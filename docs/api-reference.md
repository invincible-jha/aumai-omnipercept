# API Reference — aumai-omnipercept

Complete reference for all public classes, functions, and Pydantic models.

---

## Module: `aumai_omnipercept.models`

### `Modality`

```python
class Modality(str, Enum)
```

Enumeration of all supported input modalities.

| Member        | Value           | Description                              |
|---------------|-----------------|------------------------------------------|
| `TEXT`        | `"text"`        | Text strings                             |
| `IMAGE`       | `"image"`       | Images (file path or base64)             |
| `AUDIO`       | `"audio"`       | Audio files (WAV, MP3, FLAC)             |
| `VIDEO`       | `"video"`       | Video files (MP4, AVI, MOV)              |
| `POINT_CLOUD` | `"point_cloud"` | 3D point clouds (PLY, PCD, XYZ)          |
| `TABULAR`     | `"tabular"`     | Tabular data (CSV, Parquet)              |

---

### `FusionStrategy`

```python
class FusionStrategy(str, Enum)
```

Strategy for combining feature vectors from multiple modalities.

| Member            | Value               | Description                                                  |
|-------------------|---------------------|--------------------------------------------------------------|
| `EARLY`           | `"early"`           | Concatenate all vectors, then stride-project to target dim   |
| `LATE`            | `"late"`            | Pad to target dim, then average element-wise                 |
| `CROSS_ATTENTION` | `"cross_attention"` | Scaled dot-product attention between all modality pairs      |
| `WEIGHTED_SUM`    | `"weighted_sum"`    | Weight each modality by its L2 energy (norm squared)         |

---

### `TaskType`

```python
class TaskType(str, Enum)
```

Perception tasks that a `PerceptionPipeline` can perform.

| Member           | Value                  | Description                              |
|------------------|------------------------|------------------------------------------|
| `CLASSIFICATION` | `"classification"`     | Multi-class label prediction             |
| `DETECTION`      | `"detection"`          | Object detection with bounding boxes     |
| `SEGMENTATION`   | `"segmentation"`       | Pixel-level classification               |
| `GENERATION`     | `"generation"`         | Content generation                       |
| `RETRIEVAL`      | `"retrieval"`          | Feature extraction for similarity search |
| `CAPTIONING`     | `"captioning"`         | Natural language caption generation      |
| `QA`             | `"question_answering"` | Visual/multimodal question answering     |

---

### `ModalityInput`

```python
class ModalityInput(BaseModel)
```

A single modality input for processing by the perception pipeline.

**Fields:**

| Field               | Type              | Required | Default | Description                                                |
|---------------------|-------------------|----------|---------|------------------------------------------------------------|
| `modality`          | `Modality`        | Yes      |         | The modality type.                                         |
| `data`              | `str`             | Yes      |         | Raw text, file path, or base64-encoded bytes.             |
| `metadata`          | `dict[str, str]`  | No       | `{}`    | Arbitrary string key-value metadata pairs.                |
| `sample_rate`       | `int`             | No       | `0`     | Audio sample rate in Hz (audio modality only).            |
| `width`             | `int`             | No       | `0`     | Image or video width in pixels.                           |
| `height`            | `int`             | No       | `0`     | Image or video height in pixels.                          |
| `channels`          | `int`             | No       | `0`     | Number of color channels (3 for RGB, 1 for grayscale).    |
| `duration_seconds`  | `float`           | No       | `0.0`   | Audio or video duration in seconds.                       |

**Example:**

```python
from aumai_omnipercept import ModalityInput, Modality

text_input = ModalityInput(modality=Modality.TEXT, data="Hello world")

image_input = ModalityInput(
    modality=Modality.IMAGE,
    data="/images/photo.jpg",
    width=1920,
    height=1080,
    channels=3,
    metadata={"source": "webcam"},
)

audio_input = ModalityInput(
    modality=Modality.AUDIO,
    data="/audio/recording.wav",
    sample_rate=44100,
    duration_seconds=12.5,
)
```

---

### `FeatureVector`

```python
class FeatureVector(BaseModel)
```

An L2-normalized feature representation extracted from a single modality input.

**Fields:**

| Field                  | Type           | Constraints | Description                                       |
|------------------------|----------------|-------------|---------------------------------------------------|
| `modality`             | `Modality`     |             | The source modality.                              |
| `dimensions`           | `int`          | `>= 1`      | Length of the feature vector.                     |
| `values`               | `list[float]`  |             | The feature values (L2-normalized to unit norm).  |
| `model_name`           | `str`          |             | Name of the model used for extraction. Default: `""` |
| `extraction_time_ms`   | `float`        |             | Time taken to extract this vector in ms. Default: `0.0` |

**Properties:**

#### `norm -> float`

Computed property returning the L2 norm of `values`, rounded to 6 decimal places. For a properly normalized vector this will be approximately `1.0`.

```python
fv = extractor.extract(inp)
print(fv.norm)  # ~1.0
```

**Serialization:**

```python
json_str = fv.model_dump_json()
fv_loaded = FeatureVector.model_validate_json(json_str)
```

---

### `FusedRepresentation`

```python
class FusedRepresentation(BaseModel)
```

Combined representation produced by fusing multiple `FeatureVector` objects via `ModalityFusion`.

**Fields:**

| Field             | Type                  | Constraints | Description                                                  |
|-------------------|-----------------------|-------------|--------------------------------------------------------------|
| `modalities`      | `list[Modality]`      |             | Modalities that were fused.                                  |
| `strategy`        | `FusionStrategy`      |             | The fusion strategy that was applied.                        |
| `dimensions`      | `int`                 | `>= 1`      | Dimensionality of the fused vector.                          |
| `values`          | `list[float]`         |             | The fused feature values.                                    |
| `weights`         | `dict[str, float]`    |             | Per-modality fractional energy contribution (sums to ~1.0).  |
| `fusion_time_ms`  | `float`               |             | Time taken to perform fusion in milliseconds.                |

---

### `Prediction`

```python
class Prediction(BaseModel)
```

A single prediction from the perception pipeline.

**Fields:**

| Field        | Type              | Constraints   | Description                                                       |
|--------------|-------------------|---------------|-------------------------------------------------------------------|
| `label`      | `str`             |               | Predicted class label, object name, or caption string.           |
| `confidence` | `float`           | `>= 0, <= 1`  | Confidence score for this prediction.                             |
| `bbox`       | `list[float]`     |               | Bounding box `[x1, y1, x2, y2]` in normalized `[0, 1]` coordinates. Empty for non-detection tasks. |
| `metadata`   | `dict[str, str]`  |               | Arbitrary string metadata.                                        |

---

### `PerceptionResult`

```python
class PerceptionResult(BaseModel)
```

Result from running a full perception pipeline. Returned by `PerceptionPipeline.process()`.

**Fields:**

| Field                 | Type                | Constraints    | Description                                          |
|-----------------------|---------------------|----------------|------------------------------------------------------|
| `task`                | `TaskType`          |                | The perception task that was performed.              |
| `modalities_used`     | `list[Modality]`    |                | Modalities that contributed to the result.           |
| `predictions`         | `list[Prediction]`  |                | Ranked predictions (highest confidence first).       |
| `confidence`          | `float`             | `>= 0, <= 1`   | Average confidence across all predictions.           |
| `processing_time_ms`  | `float`             |                | Total end-to-end processing time in milliseconds.    |

---

### `PipelineConfig`

```python
class PipelineConfig(BaseModel)
```

Configuration for a `PerceptionPipeline` instance.

**Fields:**

| Field                  | Type              | Required | Constraints  | Default  | Description                                       |
|------------------------|-------------------|----------|--------------|----------|---------------------------------------------------|
| `name`                 | `str`             | Yes      |              |          | Human-readable pipeline name.                     |
| `modalities`           | `list[Modality]`  | Yes      |              |          | Accepted input modalities.                        |
| `task`                 | `TaskType`        | Yes      |              |          | Perception task to perform.                       |
| `fusion_strategy`      | `FusionStrategy`  | No       |              | `LATE`   | How to combine multi-modal feature vectors.       |
| `feature_dim`          | `int`             | No       | `>= 1`       | `256`    | Dimensionality of feature vectors.                |
| `max_predictions`      | `int`             | No       | `>= 1`       | `10`     | Maximum number of predictions to return.          |
| `confidence_threshold` | `float`           | No       | `>= 0, <= 1` | `0.5`    | Minimum confidence for a prediction to be included.|

---

### `ModelCard`

```python
class ModelCard(BaseModel)
```

Metadata describing a perception model in the registry.

**Fields:**

| Field                  | Type          | Constraints | Description                                              |
|------------------------|---------------|-------------|----------------------------------------------------------|
| `model_id`             | `str`         |             | Unique identifier string.                                |
| `name`                 | `str`         |             | Human-readable model name.                               |
| `modality`             | `Modality`    |             | The input modality this model handles.                   |
| `task`                 | `TaskType`    |             | The task this model is designed for.                     |
| `feature_dim`          | `int`         | `>= 1`      | Output feature vector dimensionality.                    |
| `parameters_millions`  | `float`       |             | Model size in millions of parameters. Default: `0.0`    |
| `description`          | `str`         |             | Human-readable description. Default: `""`               |
| `supported_formats`    | `list[str]`   |             | File format extensions this model accepts. Default: `[]` |

---

### `BenchmarkResult`

```python
class BenchmarkResult(BaseModel)
```

Evaluation results from `Benchmarker.evaluate()`.

**Fields:**

| Field               | Type       | Constraints    | Description                                                          |
|---------------------|------------|----------------|----------------------------------------------------------------------|
| `pipeline_name`     | `str`      |                | Name from the pipeline's `PipelineConfig`.                           |
| `dataset`           | `str`      |                | Dataset name passed to `evaluate()`.                                 |
| `task`              | `TaskType` |                | Task type from the pipeline's `PipelineConfig`.                      |
| `accuracy`          | `float`    | `>= 0, <= 1`   | Fraction of samples where top prediction matches ground truth.       |
| `precision`         | `float`    | `>= 0, <= 1`   | Macro-averaged precision across all labels.                          |
| `recall`            | `float`    | `>= 0, <= 1`   | Macro-averaged recall across all labels.                             |
| `f1_score`          | `float`    | `>= 0, <= 1`   | Macro-averaged F1 score (harmonic mean of precision and recall).     |
| `avg_latency_ms`    | `float`    |                | Average processing time per sample in milliseconds.                  |
| `samples_evaluated` | `int`      |                | Total number of samples evaluated.                                   |

---

## Module: `aumai_omnipercept.core`

### `ModelRegistry`

```python
class ModelRegistry
```

Registry of `ModelCard` instances. Pre-loaded with 12 built-in model cards.

#### `get(model_id: str) -> ModelCard | None`

Retrieve a model card by ID. Returns `None` if not found.

#### `by_modality(modality: Modality) -> list[ModelCard]`

Return all model cards for a given modality.

#### `by_task(task: TaskType) -> list[ModelCard]`

Return all model cards for a given task type.

#### `register(card: ModelCard) -> None`

Register a custom model card. Overwrites any existing card with the same `model_id`.

#### `all_models() -> list[ModelCard]`

Return all registered model cards.

#### `find_best(modality: Modality, task: TaskType) -> ModelCard | None`

Find the best model card for a modality and task combination. Selection: all cards matching both `modality` and `task`; falls back to `modality` only if none found; returns the card with the highest `feature_dim`. Returns `None` if no matching cards exist.

```python
from aumai_omnipercept import ModelRegistry, Modality, TaskType

registry = ModelRegistry()
best = registry.find_best(Modality.IMAGE, TaskType.RETRIEVAL)
print(f"{best.model_id}: {best.feature_dim}D")  # image-embed-v1: 512D
```

---

### `FeatureExtractor`

```python
class FeatureExtractor
```

Extracts L2-normalized feature vectors from modality inputs.

#### `__init__(self, registry: ModelRegistry | None = None) -> None`

If `registry` is `None`, a default `ModelRegistry` is created.

#### `extract(inp: ModalityInput, target_dim: int = 256) -> FeatureVector`

Extract a deterministic feature vector from the given input.

- **Parameters**:
  - `inp` — the `ModalityInput` to process
  - `target_dim` — desired feature vector length (default: `256`)
- **Returns**: `FeatureVector` with L2-normalized values, `dimensions == target_dim`

Extraction steps:
1. SHA-256 hash of `"{modality.value}:{data}"` → seed bytes
2. Expand seed via chained SHA-256 hashing to `target_dim` floats in `[-1, 1]`
3. Modality-specific modulation:
   - **Text**: dims 0-2 set to `tanh(word_count/100)`, `tanh(char_count/1000)`, `tanh(unique_words/total_words)`
   - **Image**: dims 0-2 set to `tanh(width/1920)`, `tanh(height/1080)`, `tanh(channels/4)`
   - **Audio**: dims 0-1 set to `tanh(sample_rate/48000)`, `tanh(duration_seconds/300)`
4. L2-normalize to unit norm

```python
from aumai_omnipercept import FeatureExtractor, ModalityInput, Modality

extractor = FeatureExtractor()
inp = ModalityInput(modality=Modality.TEXT, data="Hello world")
fv = extractor.extract(inp, target_dim=128)
print(fv.dimensions, fv.norm)  # 128  ~1.0
```

---

### `ModalityFusion`

```python
class ModalityFusion
```

Combines feature vectors from multiple modalities into a single fused representation.

#### `fuse(features: list[FeatureVector], strategy: FusionStrategy, target_dim: int = 256) -> FusedRepresentation`

Fuse a list of feature vectors using the specified strategy.

- **Parameters**:
  - `features` — list of `FeatureVector` objects
  - `strategy` — the `FusionStrategy` to use
  - `target_dim` — output dimensionality (default: `256`)
- **Returns**: `FusedRepresentation` with fused values, per-modality weights, and timing

If `features` is empty, returns a zero-vector `FusedRepresentation` with `fusion_time_ms=0.0`.

**Strategy algorithms:**

| Strategy          | Algorithm                                                                         |
|-------------------|-----------------------------------------------------------------------------------|
| `EARLY`           | Concatenate all `values` lists; stride-sample uniformly to `target_dim`           |
| `LATE`            | Pad each vector to `target_dim` with zeros; element-wise average                 |
| `CROSS_ATTENTION` | Each modality attends to all others via `exp(dot(Q,K)/sqrt(dim))`; weighted sum  |
| `WEIGHTED_SUM`    | Weight each vector by `sum(v^2)` (L2 energy); normalize weights to sum to 1.0   |

```python
from aumai_omnipercept import ModalityFusion, FusionStrategy

fusion = ModalityFusion()
fused = fusion.fuse([text_fv, image_fv], FusionStrategy.CROSS_ATTENTION, target_dim=256)
print(fused.weights)          # {'text': 0.4821, 'image': 0.5179}
print(fused.fusion_time_ms)   # e.g., 0.18
```

---

### `PerceptionPipeline`

```python
class PerceptionPipeline
```

End-to-end multimodal perception pipeline.

#### `__init__(self, config: PipelineConfig) -> None`

Creates the pipeline from a `PipelineConfig`. Internally instantiates `ModelRegistry`, `FeatureExtractor`, and `ModalityFusion`.

#### `process(inputs: list[ModalityInput]) -> PerceptionResult`

Run the full pipeline on a list of multimodal inputs.

Processing steps:
1. Filter inputs: keep only those whose `modality` is in `config.modalities`
2. Extract a `FeatureVector` from each valid input at `config.feature_dim` dimensions
3. If multiple features: fuse them with `config.fusion_strategy`; if single: use directly
4. Generate task-specific predictions from the combined feature values
5. Return `PerceptionResult`

- **Parameters**: `inputs` — list of `ModalityInput` objects
- **Returns**: `PerceptionResult`

**Prediction generation by task:**

| Task             | Algorithm                                                                                |
|------------------|------------------------------------------------------------------------------------------|
| `CLASSIFICATION` | First `max_predictions` feature values as logits; softmax; filter by `confidence_threshold` |
| `DETECTION`      | Features consumed in 6-value chunks `(x1, y1, dx, dy, class_idx, conf)`; normalized bboxes |
| `CAPTIONING`     | Single prediction with label describing modalities and feature energy statistics          |
| Other tasks      | Fall back to classification-style output                                                 |

```python
config = PipelineConfig(
    name="Classifier",
    modalities=[Modality.TEXT, Modality.IMAGE],
    task=TaskType.CLASSIFICATION,
    fusion_strategy=FusionStrategy.LATE,
    feature_dim=256,
    max_predictions=5,
    confidence_threshold=0.05,
)
pipeline = PerceptionPipeline(config)
result = pipeline.process([text_input, image_input])
```

---

### `SimilaritySearch`

```python
class SimilaritySearch
```

In-memory cosine similarity index for `FeatureVector` objects.

#### `add(item_id: str, feature: FeatureVector) -> None`

Add a feature vector to the index with the given item ID.

#### `search(query: FeatureVector, top_k: int = 5) -> list[tuple[str, float]]`

Find the top-k most similar items by cosine similarity.

- **Returns**: `list[tuple[str, float]]` — `(item_id, cosine_similarity)` pairs, sorted descending by score

Cosine similarity formula: `dot(a, b) / (norm(a) * norm(b))`. Returns `0.0` when either vector has near-zero norm. Linear scan — O(n) per query.

```python
from aumai_omnipercept import SimilaritySearch

index = SimilaritySearch()
index.add("item_a", fv_a)
index.add("item_b", fv_b)
results = index.search(query_fv, top_k=3)
# [("item_a", 0.9218), ("item_b", 0.7341)]
```

---

### `Benchmarker`

```python
class Benchmarker
```

Evaluates a `PerceptionPipeline` against ground truth labels.

#### `evaluate(pipeline: PerceptionPipeline, test_inputs: list[list[ModalityInput]], ground_truth: list[str], dataset_name: str = "custom") -> BenchmarkResult`

Evaluate a pipeline against a set of labeled test samples.

- **Parameters**:
  - `pipeline` — the `PerceptionPipeline` to evaluate
  - `test_inputs` — list of input lists (one list of `ModalityInput` per sample)
  - `ground_truth` — ground truth label strings (same length as `test_inputs`)
  - `dataset_name` — dataset name for reporting (default: `"custom"`)
- **Returns**: `BenchmarkResult`

Metrics:
- **Accuracy**: `correct / total`
- **Macro precision**: average precision across all unique ground truth labels
- **Macro recall**: average recall across all unique ground truth labels
- **Macro F1**: `2 * precision * recall / (precision + recall)`
- **Average latency**: `sum(processing_time_ms) / n`

```python
from aumai_omnipercept import Benchmarker

result = Benchmarker().evaluate(
    pipeline=pipeline,
    test_inputs=[[ModalityInput(modality=Modality.TEXT, data=t)] for t in test_texts],
    ground_truth=labels,
    dataset_name="eval_v1",
)
print(f"Accuracy: {result.accuracy:.4f}, F1: {result.f1_score:.4f}")
```

---

## Public Re-exports from `aumai_omnipercept`

```python
from aumai_omnipercept import (
    # Core classes
    Benchmarker,
    FeatureExtractor,
    ModalityFusion,
    ModelRegistry,
    PerceptionPipeline,
    SimilaritySearch,
    # Models
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
```

---

## Built-In Model Cards

| `model_id`           | `name`           | `modality`    | `task`             | `feature_dim` | `parameters_millions` |
|----------------------|------------------|---------------|--------------------|---------------|-----------------------|
| `text-embed-v1`      | TextEmbedder     | text          | retrieval          | 256           | 22.0                  |
| `text-classify-v1`   | TextClassifier   | text          | classification     | 128           | 15.0                  |
| `image-embed-v1`     | ImageEmbedder    | image         | retrieval          | 512           | 86.0                  |
| `image-detect-v1`    | ObjectDetector   | image         | detection          | 256           | 41.0                  |
| `image-segment-v1`   | SegmentAnything  | image         | segmentation       | 256           | 93.0                  |
| `audio-embed-v1`     | AudioEmbedder    | audio         | retrieval          | 256           | 12.0                  |
| `audio-classify-v1`  | AudioClassifier  | audio         | classification     | 128           | 8.0                   |
| `video-embed-v1`     | VideoEmbedder    | video         | retrieval          | 512           | 150.0                 |
| `video-caption-v1`   | VideoCaptioner   | video         | captioning         | 512           | 200.0                 |
| `multimodal-qa-v1`   | MultimodalQA     | text          | question_answering | 768           | 350.0                 |
| `pointcloud-v1`      | PointCloudNet    | point_cloud   | classification     | 256           | 5.0                   |
| `tabular-v1`         | TabularEncoder   | tabular       | classification     | 64            | 0.5                   |

---

## Error Reference

| Exception            | Raised By                             | Condition                                                   |
|----------------------|---------------------------------------|-------------------------------------------------------------|
| `ValidationError`    | Any Pydantic model constructor        | Invalid field types or constraint violations                 |
| `FileNotFoundError`  | CLI `fuse`, `perceive`, `search`      | Input JSON file path does not exist                         |
| `json.JSONDecodeError` | CLI commands                        | JSON file is malformed                                      |
