"""AumAI OmniPercept - Multimodal perception framework."""

__version__ = "0.1.0"

from aumai_omnipercept.core import (
    Benchmarker,
    FeatureExtractor,
    ModalityFusion,
    ModelRegistry,
    PerceptionPipeline,
    SimilaritySearch,
)
from aumai_omnipercept.models import (
    BenchmarkResult,
    FeatureVector,
    FusedRepresentation,
    FusionStrategy,
    Modality,
    ModalityInput,
    ModelCard,
    PerceptionResult,
    PipelineConfig,
    Prediction,
    TaskType,
)

__all__ = [
    "Benchmarker",
    "BenchmarkResult",
    "FeatureExtractor",
    "FeatureVector",
    "FusedRepresentation",
    "FusionStrategy",
    "Modality",
    "ModalityFusion",
    "ModalityInput",
    "ModelCard",
    "ModelRegistry",
    "PerceptionPipeline",
    "PerceptionResult",
    "PipelineConfig",
    "Prediction",
    "SimilaritySearch",
    "TaskType",
]
