"""Pydantic models for aumai-omnipercept."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    POINT_CLOUD = "point_cloud"
    TABULAR = "tabular"


class FusionStrategy(str, Enum):
    EARLY = "early"
    LATE = "late"
    CROSS_ATTENTION = "cross_attention"
    WEIGHTED_SUM = "weighted_sum"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    CAPTIONING = "captioning"
    QA = "question_answering"


class ModalityInput(BaseModel):
    """A single modality input for processing."""
    modality: Modality
    data: str  # path, raw text, or base64 for binary
    metadata: dict[str, str] = Field(default_factory=dict)
    sample_rate: int = 0  # audio only
    width: int = 0  # image/video only
    height: int = 0  # image/video only
    channels: int = 0  # image/video only
    duration_seconds: float = 0.0  # audio/video only


class FeatureVector(BaseModel):
    """Extracted feature representation from a modality."""
    modality: Modality
    dimensions: int = Field(ge=1)
    values: list[float]
    model_name: str = ""
    extraction_time_ms: float = 0.0

    @property
    def norm(self) -> float:
        return round(sum(v * v for v in self.values) ** 0.5, 6)


class FusedRepresentation(BaseModel):
    """Combined representation from multiple modalities."""
    modalities: list[Modality]
    strategy: FusionStrategy
    dimensions: int = Field(ge=1)
    values: list[float]
    weights: dict[str, float] = Field(default_factory=dict)
    fusion_time_ms: float = 0.0


class PerceptionResult(BaseModel):
    """Result from a perception task."""
    task: TaskType
    modalities_used: list[Modality]
    predictions: list[Prediction] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    processing_time_ms: float = 0.0


class Prediction(BaseModel):
    """A single prediction from the perception pipeline."""
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(default_factory=list)  # [x1, y1, x2, y2] for detection
    metadata: dict[str, str] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Configuration for a perception pipeline."""
    name: str
    modalities: list[Modality]
    task: TaskType
    fusion_strategy: FusionStrategy = FusionStrategy.LATE
    feature_dim: int = Field(default=256, ge=1)
    max_predictions: int = Field(default=10, ge=1)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)


class ModelCard(BaseModel):
    """Metadata about a perception model."""
    model_id: str
    name: str
    modality: Modality
    task: TaskType
    feature_dim: int = Field(ge=1)
    parameters_millions: float = 0.0
    description: str = ""
    supported_formats: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    """Benchmark result for a perception pipeline."""
    pipeline_name: str
    dataset: str
    task: TaskType
    accuracy: float = Field(ge=0, le=1, default=0.0)
    precision: float = Field(ge=0, le=1, default=0.0)
    recall: float = Field(ge=0, le=1, default=0.0)
    f1_score: float = Field(ge=0, le=1, default=0.0)
    avg_latency_ms: float = 0.0
    samples_evaluated: int = 0


# Fix forward reference
PerceptionResult.model_rebuild()


__all__ = [
    "Modality", "FusionStrategy", "TaskType",
    "ModalityInput", "FeatureVector", "FusedRepresentation",
    "PerceptionResult", "Prediction", "PipelineConfig",
    "ModelCard", "BenchmarkResult",
]
