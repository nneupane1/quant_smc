__all__ = ["__version__", "ConfigLoader", "TrainOrchestrator"]

__version__ = "1.0.0"


def __getattr__(name):
    if name == "ConfigLoader":
        from quant_system.config.config_loader import ConfigLoader

        return ConfigLoader
    if name == "TrainOrchestrator":
        from quant_system.train_orchestrator import TrainOrchestrator

        return TrainOrchestrator
    raise AttributeError(f"module 'quant_system' has no attribute {name!r}")
