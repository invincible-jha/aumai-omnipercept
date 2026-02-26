"""Core logic for aumai-omnipercept."""

from __future__ import annotations

import hashlib
import math
import time

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

_MODEL_REGISTRY: list[ModelCard] = [
    ModelCard(model_id="text-embed-v1", name="TextEmbedder", modality=Modality.TEXT,
              task=TaskType.RETRIEVAL, feature_dim=256, parameters_millions=22.0,
              description="Lightweight text embedding model for retrieval tasks.",
              supported_formats=["txt", "json", "csv"]),
    ModelCard(model_id="text-classify-v1", name="TextClassifier", modality=Modality.TEXT,
              task=TaskType.CLASSIFICATION, feature_dim=128, parameters_millions=15.0,
              description="Text classification model for sentiment, topic, intent.",
              supported_formats=["txt", "json"]),
    ModelCard(model_id="image-embed-v1", name="ImageEmbedder", modality=Modality.IMAGE,
              task=TaskType.RETRIEVAL, feature_dim=512, parameters_millions=86.0,
              description="Vision transformer for image feature extraction.",
              supported_formats=["jpg", "png", "bmp", "webp"]),
    ModelCard(model_id="image-detect-v1", name="ObjectDetector", modality=Modality.IMAGE,
              task=TaskType.DETECTION, feature_dim=256, parameters_millions=41.0,
              description="Object detection model with bounding box predictions.",
              supported_formats=["jpg", "png"]),
    ModelCard(model_id="image-segment-v1", name="SegmentAnything", modality=Modality.IMAGE,
              task=TaskType.SEGMENTATION, feature_dim=256, parameters_millions=93.0,
              description="Image segmentation model for pixel-level classification.",
              supported_formats=["jpg", "png"]),
    ModelCard(model_id="audio-embed-v1", name="AudioEmbedder", modality=Modality.AUDIO,
              task=TaskType.RETRIEVAL, feature_dim=256, parameters_millions=12.0,
              description="Audio feature extraction using mel-spectrogram + transformer.",
              supported_formats=["wav", "mp3", "flac"]),
    ModelCard(model_id="audio-classify-v1", name="AudioClassifier", modality=Modality.AUDIO,
              task=TaskType.CLASSIFICATION, feature_dim=128, parameters_millions=8.0,
              description="Audio event classification and speech detection.",
              supported_formats=["wav", "mp3"]),
    ModelCard(model_id="video-embed-v1", name="VideoEmbedder", modality=Modality.VIDEO,
              task=TaskType.RETRIEVAL, feature_dim=512, parameters_millions=150.0,
              description="Video understanding model with temporal modeling.",
              supported_formats=["mp4", "avi", "mov"]),
    ModelCard(model_id="video-caption-v1", name="VideoCaptioner", modality=Modality.VIDEO,
              task=TaskType.CAPTIONING, feature_dim=512, parameters_millions=200.0,
              description="Video captioning with temporal attention.",
              supported_formats=["mp4"]),
    ModelCard(model_id="multimodal-qa-v1", name="MultimodalQA", modality=Modality.TEXT,
              task=TaskType.QA, feature_dim=768, parameters_millions=350.0,
              description="Visual question answering with cross-modal attention.",
              supported_formats=["txt", "jpg", "png"]),
    ModelCard(model_id="pointcloud-v1", name="PointCloudNet", modality=Modality.POINT_CLOUD,
              task=TaskType.CLASSIFICATION, feature_dim=256, parameters_millions=5.0,
              description="3D point cloud classification and segmentation.",
              supported_formats=["ply", "pcd", "xyz"]),
    ModelCard(model_id="tabular-v1", name="TabularEncoder", modality=Modality.TABULAR,
              task=TaskType.CLASSIFICATION, feature_dim=64, parameters_millions=0.5,
              description="Tabular data encoding with learned embeddings.",
              supported_formats=["csv", "parquet"]),
]


class ModelRegistry:
    """Registry of available perception models."""

    def __init__(self) -> None:
        self._models: dict[str, ModelCard] = {m.model_id: m for m in _MODEL_REGISTRY}

    def get(self, model_id: str) -> ModelCard | None:
        return self._models.get(model_id)

    def by_modality(self, modality: Modality) -> list[ModelCard]:
        return [m for m in self._models.values() if m.modality == modality]

    def by_task(self, task: TaskType) -> list[ModelCard]:
        return [m for m in self._models.values() if m.task == task]

    def register(self, card: ModelCard) -> None:
        self._models[card.model_id] = card

    def all_models(self) -> list[ModelCard]:
        return list(self._models.values())

    def find_best(self, modality: Modality, task: TaskType) -> ModelCard | None:
        """Find best model for a given modality and task combination."""
        candidates = [m for m in self._models.values() if m.modality == modality and m.task == task]
        if not candidates:
            candidates = [m for m in self._models.values() if m.modality == modality]
        return max(candidates, key=lambda m: m.feature_dim) if candidates else None


class FeatureExtractor:
    """Extract feature vectors from modality inputs."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or ModelRegistry()

    def extract(self, inp: ModalityInput, target_dim: int = 256) -> FeatureVector:
        """Extract a deterministic feature vector from input data.

        Uses hash-based feature generation for reproducible results without
        requiring actual ML model inference.
        """
        start = time.monotonic()
        # Deterministic feature generation from input data
        seed = hashlib.sha256(f"{inp.modality.value}:{inp.data}".encode()).digest()
        values = self._hash_to_features(seed, target_dim)

        # Apply modality-specific transformations
        if inp.modality == Modality.TEXT:
            values = self._text_features(inp.data, values)
        elif inp.modality == Modality.IMAGE:
            values = self._image_features(inp, values)
        elif inp.modality == Modality.AUDIO:
            values = self._audio_features(inp, values)

        # L2 normalize
        values = self._normalize(values)

        elapsed = (time.monotonic() - start) * 1000
        model = self._registry.find_best(inp.modality, TaskType.RETRIEVAL)
        return FeatureVector(
            modality=inp.modality,
            dimensions=target_dim,
            values=values,
            model_name=model.name if model else "hash-based",
            extraction_time_ms=round(elapsed, 2),
        )

    def _hash_to_features(self, seed: bytes, dim: int) -> list[float]:
        """Generate deterministic float values from a hash seed."""
        values: list[float] = []
        current = seed
        while len(values) < dim:
            current = hashlib.sha256(current).digest()
            for i in range(0, len(current) - 3, 4):
                if len(values) >= dim:
                    break
                # Convert 4 bytes to float in [-1, 1]
                raw = int.from_bytes(current[i:i + 4], "big")
                values.append((raw / 2147483648.0) - 1.0)
        return values[:dim]

    def _text_features(self, text: str, base: list[float]) -> list[float]:
        """Enhance features with text-specific signals."""
        word_count = len(text.split())
        char_count = len(text)
        # Modulate first few dimensions with text statistics
        if len(base) > 3:
            base[0] = math.tanh(word_count / 100)
            base[1] = math.tanh(char_count / 1000)
            base[2] = math.tanh(len(set(text.lower().split())) / max(word_count, 1))
        return base

    def _image_features(self, inp: ModalityInput, base: list[float]) -> list[float]:
        """Enhance features with image metadata signals."""
        if len(base) > 3 and inp.width > 0 and inp.height > 0:
            base[0] = math.tanh(inp.width / 1920)
            base[1] = math.tanh(inp.height / 1080)
            base[2] = math.tanh(inp.channels / 4)
        return base

    def _audio_features(self, inp: ModalityInput, base: list[float]) -> list[float]:
        """Enhance features with audio metadata signals."""
        if len(base) > 2 and inp.sample_rate > 0:
            base[0] = math.tanh(inp.sample_rate / 48000)
            base[1] = math.tanh(inp.duration_seconds / 300)
        return base

    def _normalize(self, values: list[float]) -> list[float]:
        """L2 normalize the feature vector."""
        norm = math.sqrt(sum(v * v for v in values))
        if norm < 1e-10:
            return values
        return [round(v / norm, 8) for v in values]


class ModalityFusion:
    """Fuse features from multiple modalities."""

    def fuse(self, features: list[FeatureVector], strategy: FusionStrategy, target_dim: int = 256) -> FusedRepresentation:
        """Combine multiple feature vectors into a fused representation."""
        start = time.monotonic()

        if not features:
            return FusedRepresentation(
                modalities=[], strategy=strategy, dimensions=target_dim,
                values=[0.0] * target_dim, fusion_time_ms=0.0,
            )

        if strategy == FusionStrategy.EARLY:
            values = self._early_fusion(features, target_dim)
        elif strategy == FusionStrategy.LATE:
            values = self._late_fusion(features, target_dim)
        elif strategy == FusionStrategy.CROSS_ATTENTION:
            values = self._cross_attention_fusion(features, target_dim)
        elif strategy == FusionStrategy.WEIGHTED_SUM:
            values = self._weighted_sum_fusion(features, target_dim)
        else:
            values = self._late_fusion(features, target_dim)

        # Calculate per-modality contribution weights
        weights: dict[str, float] = {}
        for fv in features:
            energy = sum(v * v for v in fv.values)
            weights[fv.modality.value] = round(energy, 4)
        total_energy = sum(weights.values())
        if total_energy > 0:
            weights = {k: round(v / total_energy, 4) for k, v in weights.items()}

        elapsed = (time.monotonic() - start) * 1000
        return FusedRepresentation(
            modalities=[f.modality for f in features],
            strategy=strategy,
            dimensions=target_dim,
            values=values,
            weights=weights,
            fusion_time_ms=round(elapsed, 2),
        )

    def _early_fusion(self, features: list[FeatureVector], target_dim: int) -> list[float]:
        """Concatenate then project to target dimension."""
        concat: list[float] = []
        for fv in features:
            concat.extend(fv.values)
        # Simple linear projection via stride sampling
        if len(concat) <= target_dim:
            return concat + [0.0] * (target_dim - len(concat))
        stride = len(concat) / target_dim
        return [concat[min(int(i * stride), len(concat) - 1)] for i in range(target_dim)]

    def _late_fusion(self, features: list[FeatureVector], target_dim: int) -> list[float]:
        """Average aligned feature vectors."""
        result = [0.0] * target_dim
        for fv in features:
            padded = fv.values[:target_dim] + [0.0] * max(0, target_dim - len(fv.values))
            for i in range(target_dim):
                result[i] += padded[i]
        n = len(features)
        return [round(v / n, 8) for v in result]

    def _cross_attention_fusion(self, features: list[FeatureVector], target_dim: int) -> list[float]:
        """Simulated cross-attention: each modality attends to all others."""
        if len(features) < 2:
            return self._late_fusion(features, target_dim)

        result = [0.0] * target_dim
        for i, query in enumerate(features):
            # Compute attention scores against all other features
            scores: list[float] = []
            for j, key in enumerate(features):
                if i == j:
                    continue
                # Dot product attention
                dim = min(len(query.values), len(key.values), target_dim)
                dot = sum(query.values[k] * key.values[k] for k in range(dim))
                scores.append(math.exp(dot / math.sqrt(dim)))
            total_score = sum(scores) or 1.0
            weights = [s / total_score for s in scores]

            # Weighted sum of key features
            idx = 0
            for j, key in enumerate(features):
                if i == j:
                    continue
                w = weights[idx]
                padded = key.values[:target_dim] + [0.0] * max(0, target_dim - len(key.values))
                for k in range(target_dim):
                    result[k] += w * padded[k]
                idx += 1

        n = len(features)
        return [round(v / n, 8) for v in result]

    def _weighted_sum_fusion(self, features: list[FeatureVector], target_dim: int) -> list[float]:
        """Weight each modality by its feature energy (L2 norm squared)."""
        energies = [sum(v * v for v in fv.values) for fv in features]
        total = sum(energies) or 1.0
        weights = [e / total for e in energies]

        result = [0.0] * target_dim
        for fv, w in zip(features, weights):
            padded = fv.values[:target_dim] + [0.0] * max(0, target_dim - len(fv.values))
            for i in range(target_dim):
                result[i] += w * padded[i]
        return [round(v, 8) for v in result]


class PerceptionPipeline:
    """End-to-end multimodal perception pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._registry = ModelRegistry()
        self._extractor = FeatureExtractor(self._registry)
        self._fusion = ModalityFusion()

    def process(self, inputs: list[ModalityInput]) -> PerceptionResult:
        """Run full perception pipeline on multimodal inputs."""
        start = time.monotonic()

        # Validate modalities
        valid_inputs = [inp for inp in inputs if inp.modality in self.config.modalities]

        # Extract features
        features: list[FeatureVector] = []
        for inp in valid_inputs:
            fv = self._extractor.extract(inp, self.config.feature_dim)
            features.append(fv)

        # Fuse if multiple modalities
        if len(features) > 1:
            fused = self._fusion.fuse(features, self.config.fusion_strategy, self.config.feature_dim)
            combined_values = fused.values
        elif features:
            combined_values = features[0].values
        else:
            combined_values = [0.0] * self.config.feature_dim

        # Generate predictions from fused representation
        predictions = self._generate_predictions(combined_values, valid_inputs)

        elapsed = (time.monotonic() - start) * 1000
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions) if predictions else 0.0

        return PerceptionResult(
            task=self.config.task,
            modalities_used=[inp.modality for inp in valid_inputs],
            predictions=predictions,
            confidence=round(avg_confidence, 4),
            processing_time_ms=round(elapsed, 2),
        )

    def _generate_predictions(self, values: list[float], inputs: list[ModalityInput]) -> list[Prediction]:
        """Generate task-specific predictions from fused features."""
        if self.config.task == TaskType.CLASSIFICATION:
            return self._classify(values)
        if self.config.task == TaskType.DETECTION:
            return self._detect(values)
        if self.config.task == TaskType.CAPTIONING:
            return self._caption(values, inputs)
        # Default: classification-style output
        return self._classify(values)

    def _classify(self, values: list[float]) -> list[Prediction]:
        """Generate classification predictions from feature values."""
        # Use first N feature values as logits for N classes
        n_classes = min(self.config.max_predictions, len(values))
        logits = values[:n_classes]
        # Softmax
        max_logit = max(logits) if logits else 0
        exps = [math.exp(v - max_logit) for v in logits]
        total = sum(exps) or 1.0
        probs = [e / total for e in exps]

        predictions = [
            Prediction(label=f"class_{i}", confidence=round(p, 4))
            for i, p in enumerate(probs)
            if p >= self.config.confidence_threshold
        ]
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:self.config.max_predictions]

    def _detect(self, values: list[float]) -> list[Prediction]:
        """Generate detection predictions with bounding boxes."""
        predictions: list[Prediction] = []
        # Each detection uses 6 values: x1, y1, x2, y2, class_idx, confidence
        chunk_size = 6
        for i in range(0, min(len(values), self.config.max_predictions * chunk_size), chunk_size):
            if i + chunk_size > len(values):
                break
            chunk = values[i:i + chunk_size]
            # Normalize bbox coords to [0, 1]
            x1 = abs(chunk[0]) % 1.0
            y1 = abs(chunk[1]) % 1.0
            x2 = min(x1 + abs(chunk[2]) % 0.5 + 0.05, 1.0)
            y2 = min(y1 + abs(chunk[3]) % 0.5 + 0.05, 1.0)
            conf = abs(math.tanh(chunk[5]))
            if conf >= self.config.confidence_threshold:
                predictions.append(Prediction(
                    label=f"object_{int(abs(chunk[4]) * 10) % 20}",
                    confidence=round(conf, 4),
                    bbox=[round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                ))
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:self.config.max_predictions]

    def _caption(self, values: list[float], inputs: list[ModalityInput]) -> list[Prediction]:
        """Generate a caption prediction."""
        # Deterministic caption from feature statistics
        mean_val = sum(values) / len(values) if values else 0
        energy = sum(v * v for v in values)
        modalities = ", ".join(inp.modality.value for inp in inputs)
        caption = f"Multimodal content ({modalities}) with feature energy {energy:.2f}"
        return [Prediction(label=caption, confidence=round(abs(math.tanh(mean_val)), 4))]


class SimilaritySearch:
    """Search for similar items using feature vectors."""

    def __init__(self) -> None:
        self._index: list[tuple[str, FeatureVector]] = []

    def add(self, item_id: str, feature: FeatureVector) -> None:
        self._index.append((item_id, feature))

    def search(self, query: FeatureVector, top_k: int = 5) -> list[tuple[str, float]]:
        """Find top-k most similar items by cosine similarity."""
        results: list[tuple[str, float]] = []
        for item_id, stored in self._index:
            sim = self._cosine_similarity(query.values, stored.values)
            results.append((item_id, round(sim, 6)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dim = min(len(a), len(b))
        dot = sum(a[i] * b[i] for i in range(dim))
        norm_a = math.sqrt(sum(v * v for v in a[:dim]))
        norm_b = math.sqrt(sum(v * v for v in b[:dim]))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)


class Benchmarker:
    """Benchmark perception pipelines."""

    def evaluate(self, pipeline: PerceptionPipeline, test_inputs: list[list[ModalityInput]],
                 ground_truth: list[str], dataset_name: str = "custom") -> BenchmarkResult:
        """Evaluate a pipeline against ground truth labels."""
        correct = 0
        total_time = 0.0
        true_positives: dict[str, int] = {}
        false_positives: dict[str, int] = {}
        false_negatives: dict[str, int] = {}

        for inputs, gt_label in zip(test_inputs, ground_truth):
            result = pipeline.process(inputs)
            total_time += result.processing_time_ms
            predicted = result.predictions[0].label if result.predictions else ""
            if predicted == gt_label:
                correct += 1
                true_positives[gt_label] = true_positives.get(gt_label, 0) + 1
            else:
                false_positives[predicted] = false_positives.get(predicted, 0) + 1
                false_negatives[gt_label] = false_negatives.get(gt_label, 0) + 1

        n = len(ground_truth)
        accuracy = correct / n if n > 0 else 0.0

        # Macro precision/recall/F1
        labels = set(ground_truth)
        precisions: list[float] = []
        recalls: list[float] = []
        for label in labels:
            tp = true_positives.get(label, 0)
            fp = false_positives.get(label, 0)
            fn = false_negatives.get(label, 0)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)

        avg_p = sum(precisions) / len(precisions) if precisions else 0.0
        avg_r = sum(recalls) / len(recalls) if recalls else 0.0
        f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

        return BenchmarkResult(
            pipeline_name=pipeline.config.name,
            dataset=dataset_name,
            task=pipeline.config.task,
            accuracy=round(accuracy, 4),
            precision=round(avg_p, 4),
            recall=round(avg_r, 4),
            f1_score=round(f1, 4),
            avg_latency_ms=round(total_time / max(n, 1), 2),
            samples_evaluated=n,
        )
