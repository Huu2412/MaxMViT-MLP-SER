
import logging
import os
import copy
import random
from torch.utils.data import DataLoader, DistributedSampler
from .ravdess import get_ravdess_dataloaders, get_ravdess_test_loader, RAVDESSHFDataset
from .visec import get_visec_dataloaders, ViSECDataset, CachedViSECDataset

def get_dataloaders(config):
    """
    Factory method to get dataloaders based on config.

    Supports cross-dataset evaluation:
      If dataset.test_on == "ravdess", train loader comes from ViSEC and
      val/test loader comes from RAVDESS (filtered to shared emotion classes).
    """
    ds_config = config.get('dataset', {})
    name = ds_config.get('name', '').lower()
    test_on = ds_config.get('test_on', None)  # e.g. "ravdess" for cross-dataset eval

    batch_size = ds_config.get('args', {}).get('batch_size', 16)
    num_workers = ds_config.get('args', {}).get('num_workers', 4)
    hf_id = ds_config.get('args', {}).get('hf_id', '')
    ravdess_hf_id = ds_config.get('args', {}).get('ravdess_hf_id', 'TwinkStart/RAVDESS')

    # Augmentation configurations
    spec_augment_cfg = ds_config.get('args', {}).get('spec_augment', None)
    pitch_shift_cfg = ds_config.get('args', {}).get('pitch_shift', None)
    time_shift_cfg = ds_config.get('args', {}).get('time_shift', None)
    waveform_augment_cfg = ds_config.get('args', {}).get('waveform_augment', None)
    cache_dir = ds_config.get('args', {}).get('cache_dir', None)
    target_size = tuple(ds_config.get('args', {}).get('target_size', [224, 224]))

    # Extract seed from training config (default to 42)
    train_cfg = config.get('training', {})
    seed = train_cfg.get('seed', 42)

    # Auxiliary task config (Region Recognition)
    aux_cfg = config.get('auxiliary_task', {})
    load_accent = aux_cfg.get('enabled', False) and aux_cfg.get('task', '') == 'accent'

    logging.info(f"Factory initializing dataset: {name} with split seed: {seed}")
    if test_on:
        logging.info(f"Cross-dataset evaluation: train={name.upper()}, test={test_on.upper()}")
    if load_accent:
        logging.info("Auxiliary task: Accent/Region Recognition ENABLED")

    # ─── Cross-dataset: Train on ViSEC, Test on RAVDESS ───
    if name in ['visec', 'anyf'] and test_on == 'ravdess':
        if not hf_id: hf_id = "hustep-lab/ViSEC"

        # Shared label space (ViSEC order)
        shared_classes = ['happy', 'neutral', 'sad', 'angry']
        shared_class_map = {c: i for i, c in enumerate(shared_classes)}

        logging.info(f"Shared emotion classes: {shared_class_map}")

        # Train loader: ViSEC (train split, with augmentation)
        train_loader, _ = get_visec_dataloaders(
            hf_id=hf_id,
            batch_size=batch_size,
            num_workers=num_workers,
            spec_augment_cfg=spec_augment_cfg,
            pitch_shift_cfg=pitch_shift_cfg,
            time_shift_cfg=time_shift_cfg,
            seed=seed,
            load_accent=load_accent,
            waveform_augment_cfg=waveform_augment_cfg,
            cache_dir=cache_dir,
            target_size=target_size
        )

        # Test loader: RAVDESS (full set, no augmentation, filtered to shared classes)
        test_loader = get_ravdess_test_loader(
            hf_id=ravdess_hf_id,
            batch_size=batch_size,
            num_workers=num_workers,
            target_classes=shared_classes,
            class_map=shared_class_map
        )

        return train_loader, test_loader

    # ─── Standard: RAVDESS only ───
    elif name == 'ravdess':
        if not hf_id: hf_id = "TwinkStart/RAVDESS"
        return get_ravdess_dataloaders(
            hf_id=hf_id,
            batch_size=batch_size,
            num_workers=num_workers,
            spec_augment_cfg=spec_augment_cfg,
            pitch_shift_cfg=pitch_shift_cfg,
            time_shift_cfg=time_shift_cfg,
            seed=seed
        )

    # ─── Standard: ViSEC only ───
    elif name in ['visec', 'anyf']:
        if not hf_id: hf_id = "hustep-lab/ViSEC"
        return get_visec_dataloaders(
            hf_id=hf_id,
            batch_size=batch_size,
            num_workers=num_workers,
            spec_augment_cfg=spec_augment_cfg,
            pitch_shift_cfg=pitch_shift_cfg,
            time_shift_cfg=time_shift_cfg,
            seed=seed,
            load_accent=load_accent,
            waveform_augment_cfg=waveform_augment_cfg,
            cache_dir=cache_dir,
            target_size=target_size
        )

    else:
        raise ValueError(f"Unknown dataset name: {name}. Supported: ravdess, visec, anyf")


def _build_visec_datasets(config):
    """
    Build ViSEC train/val Dataset objects WITHOUT wrapping in DataLoader.
    Reuses the same splitting logic as get_visec_dataloaders.
    
    Returns:
        (train_ds, val_ds) or (train_ds, None) if val not applicable.
    """
    ds_config = config.get('dataset', {})
    hf_id = ds_config.get('args', {}).get('hf_id', 'hustep-lab/ViSEC')
    spec_augment_cfg = ds_config.get('args', {}).get('spec_augment', None)
    pitch_shift_cfg = ds_config.get('args', {}).get('pitch_shift', None)
    time_shift_cfg = ds_config.get('args', {}).get('time_shift', None)
    waveform_augment_cfg = ds_config.get('args', {}).get('waveform_augment', None)
    seed = config.get('training', {}).get('seed', 42)
    
    aux_cfg = config.get('auxiliary_task', {})
    load_accent = aux_cfg.get('enabled', False) and aux_cfg.get('task', '') == 'accent'
    cache_dir = ds_config.get('args', {}).get('cache_dir', None)
    target_size = tuple(ds_config.get('args', {}).get('target_size', [224, 224]))

    target_cache = cache_dir or ("visec_features" if os.path.exists(os.path.join("visec_features", "metadata.pkl")) else None)
    if target_cache and os.path.exists(os.path.join(target_cache, "metadata.pkl")):
        print(f"[Fast I/O] DDP Building datasets from cache: {target_cache}...")
        split_file = os.path.join(target_cache, "split_indices.pkl")
        train_indices, val_indices = None, None
        if os.path.exists(split_file):
            import pickle
            try:
                with open(split_file, 'rb') as f:
                    splits = pickle.load(f)
                    if splits.get('seed') == seed:
                        train_indices = splits.get('train')
                        val_indices = splits.get('val')
                    else:
                        print(f"[Seed Split] Config seed ({seed}) differs from cache seed ({splits.get('seed')}). Dynamically splitting with seed={seed}...")
            except Exception as e:
                print(f"Warning reading split_indices: {e}")

        if train_indices is None or val_indices is None:
            import pickle
            with open(os.path.join(target_cache, "metadata.pkl"), 'rb') as f:
                all_meta = pickle.load(f)
            all_idx = list(range(len(all_meta)))
            rng = random.Random(seed)
            rng.shuffle(all_idx)
            val_len = int(len(all_idx) * 0.2)
            train_indices = all_idx[val_len:]
            val_indices = all_idx[:val_len]

        train_ds = CachedViSECDataset(
            cache_dir=target_cache,
            split_indices=train_indices,
            augment=True,
            spec_augment_cfg=spec_augment_cfg,
            target_size=target_size,
            load_accent=load_accent
        )
        val_ds = CachedViSECDataset(
            cache_dir=target_cache,
            split_indices=val_indices,
            augment=False,
            target_size=target_size,
            load_accent=load_accent
        )
        print(f"Split complete. Train: {len(train_ds)}, Val: {len(val_ds)}")
        return train_ds, val_ds

    local_train_csv = os.path.join("visec_dataset", "train.csv")
    local_val_csv = os.path.join("visec_dataset", "val.csv")
    
    if os.path.exists(local_train_csv) and os.path.exists(local_val_csv):
        print(f"Loading local preprocessed datasets from {local_train_csv} and {local_val_csv}...")
        train_ds = ViSECDataset(
            hf_id=hf_id,
            spec_augment_cfg=spec_augment_cfg,
            pitch_shift_cfg=pitch_shift_cfg,
            time_shift_cfg=time_shift_cfg,
            load_accent=load_accent,
            csv_path=local_train_csv,
            augment=True,
            waveform_augment_cfg=waveform_augment_cfg
        )
        val_ds = ViSECDataset(
            hf_id=hf_id,
            spec_augment_cfg=spec_augment_cfg,
            pitch_shift_cfg=pitch_shift_cfg,
            time_shift_cfg=time_shift_cfg,
            load_accent=load_accent,
            csv_path=local_val_csv,
            augment=False,
            waveform_augment_cfg=waveform_augment_cfg
        )
    else:
        dataset = ViSECDataset(
            hf_id=hf_id, 
            spec_augment_cfg=spec_augment_cfg, 
            pitch_shift_cfg=pitch_shift_cfg, 
            time_shift_cfg=time_shift_cfg,
            load_accent=load_accent,
            waveform_augment_cfg=waveform_augment_cfg
        )
        
        full_indices = dataset.indices
        total_len = len(full_indices)
        val_len = int(total_len * 0.2)
        train_len = total_len - val_len
        
        rng = random.Random(seed)
        rng.shuffle(full_indices)
        
        train_indices = full_indices[:train_len]
        val_indices = full_indices[train_len:]
        
        train_ds = copy.copy(dataset)
        train_ds.indices = train_indices
        train_ds.augment = True
        
        val_ds = copy.copy(dataset)
        val_ds.indices = val_indices
        val_ds.augment = False
    
    print(f"Split complete. Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds


def get_dataloaders_ddp(config, rank, world_size):
    """
    Factory method for DDP (DistributedDataParallel) training.
    
    Builds the same datasets as get_dataloaders, but wraps the training
    DataLoader with a DistributedSampler so each GPU processes a distinct
    shard of the training data.
    
    Validation/test loaders run the FULL dataset on every rank (no sampling)
    to ensure consistent metric computation.
    
    Args:
        config: Full config dict
        rank: Current process rank (from dist.get_rank())
        world_size: Total number of processes (from dist.get_world_size())
    
    Returns:
        (train_loader, val_loader, train_sampler)
        train_sampler is returned so the caller can call set_epoch() each epoch.
    """
    ds_config = config.get('dataset', {})
    name = ds_config.get('name', '').lower()
    test_on = ds_config.get('test_on', None)
    
    batch_size = ds_config.get('args', {}).get('batch_size', 16)
    num_workers = ds_config.get('args', {}).get('num_workers', 4)
    ravdess_hf_id = ds_config.get('args', {}).get('ravdess_hf_id', 'TwinkStart/RAVDESS')
    
    try:
        # ─── Cross-dataset: Train on ViSEC, Test on RAVDESS ───
        if name in ['visec', 'anyf'] and test_on == 'ravdess':
            shared_classes = ['happy', 'neutral', 'sad', 'angry']
            shared_class_map = {c: i for i, c in enumerate(shared_classes)}
            
            train_ds, _ = _build_visec_datasets(config)
            
            # DistributedSampler for training
            train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True
            )
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=train_sampler,
                num_workers=num_workers, drop_last=True, pin_memory=True
            )
            
            # Test on RAVDESS — full dataset, no distributed sampling
            test_loader = get_ravdess_test_loader(
                hf_id=ravdess_hf_id,
                batch_size=batch_size,
                num_workers=num_workers,
                target_classes=shared_classes,
                class_map=shared_class_map
            )
            
            return train_loader, test_loader, train_sampler
        
        # ─── Standard: ViSEC only ───
        elif name in ['visec', 'anyf']:
            train_ds, val_ds = _build_visec_datasets(config)
            
            train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True
            )
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=train_sampler,
                num_workers=num_workers, drop_last=True, pin_memory=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
            
            return train_loader, val_loader, train_sampler
        
        # ─── Standard: RAVDESS only ───
        elif name == 'ravdess':
            hf_id = ds_config.get('args', {}).get('hf_id', 'TwinkStart/RAVDESS')
            spec_augment_cfg = ds_config.get('args', {}).get('spec_augment', None)
            pitch_shift_cfg = ds_config.get('args', {}).get('pitch_shift', None)
            time_shift_cfg = ds_config.get('args', {}).get('time_shift', None)
            seed = config.get('training', {}).get('seed', 42)
            
            full_ds = RAVDESSHFDataset(
                hf_id, split="ravdess_emo",
                spec_augment_cfg=spec_augment_cfg,
                pitch_shift_cfg=pitch_shift_cfg,
                time_shift_cfg=time_shift_cfg
            )
            
            full_indices = full_ds.indices
            total = len(full_indices)
            val_len = int(total * 0.2)
            train_len = total - val_len
            
            rng = random.Random(seed)
            rng.shuffle(full_indices)
            
            train_indices = full_indices[:train_len]
            val_indices = full_indices[train_len:]
            
            train_ds = copy.deepcopy(full_ds)
            train_ds.indices = train_indices
            train_ds.augment = True
            
            val_ds = copy.deepcopy(full_ds)
            val_ds.indices = val_indices
            val_ds.augment = False
            
            train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True
            )
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=train_sampler,
                num_workers=num_workers, drop_last=True, pin_memory=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
            
            return train_loader, val_loader, train_sampler
        
        else:
            raise ValueError(f"Unknown dataset name: {name}. Supported: ravdess, visec, anyf")
    
    except Exception as e:
        print(f"DDP dataset load error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

