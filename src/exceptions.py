class HittingAnalysisError(Exception):
    """Base exception for the JMJ Hitting Analysis pipeline."""


class VideoProcessingError(HittingAnalysisError):
    """Raised when a video cannot be opened, read, or has invalid metadata."""


class ModelInferenceError(HittingAnalysisError):
    """Raised when a model inference fails unexpectedly."""


class ConfigError(HittingAnalysisError):
    """Raised when channel configuration is missing or malformed."""


class PipelineError(HittingAnalysisError):
    """Raised when pipeline orchestration fails (e.g., corrupt intermediate JSON)."""
