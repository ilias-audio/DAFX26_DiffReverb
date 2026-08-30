"""Training utilities.

    utils            normalization constants, RT/response maps
    feature_cache    per-step cross-loss feature cache
    losses           loss classes (dasp imports are lazy)
    loss_registry    short name -> loss factory, presets
    lightning_module ShellLightningModule and WeightedCriterion
    datamodule       single (impulse, target) DataModule
    metrics          ReverbAnalyzer and compare_rir_metrics

Submodules are imported by callers, not here, so this file stays light.
"""

__all__ = [
    "utils", "feature_cache", "losses", "loss_registry",
    "lightning_module", "datamodule", "metrics",
]
