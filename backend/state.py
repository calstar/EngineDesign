"""In-memory application state for config and runner."""

from typing import Optional
from engine.pipeline.config_schemas import PintleEngineConfig
from engine.core.runner import PintleEngineRunner


class AppState:
    """Holds the current config and runner instance."""
    
    def __init__(self):
        self.config: Optional[PintleEngineConfig] = None
        self.runner: Optional[PintleEngineRunner] = None
    
    def set_config(self, config: PintleEngineConfig) -> None:
        """Set config and create a new runner instance."""
        self.config = config
        self.runner = PintleEngineRunner(config)
    
    def has_config(self) -> bool:
        """Check if a config is loaded."""
        return self.config is not None and self.runner is not None


# Global singleton instance
app_state = AppState()

