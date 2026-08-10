
import logging
from .ravdess import get_ravdess_dataloaders, get_ravdess_test_loader
from .visec import get_visec_dataloaders

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
            waveform_augment_cfg=waveform_augment_cfg
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
            waveform_augment_cfg=waveform_augment_cfg
        )

    else:
        raise ValueError(f"Unknown dataset name: {name}. Supported: ravdess, visec, anyf")
