# Getting Started with aumai-omnipercept

This guide walks you from zero to a working multimodal perception pipeline in under ten minutes. No GPU required. No model weights to download.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Uses modern type hint syntax |
| pip | 23+ | Or uv / poetry |
| pydantic | v2 | Installed automatically as a dependency |
| click | 8+ | Installed automatically as a dependency |

Optional but useful: `jq` for inspecting JSON config files, `ipython` for interactive exploration.

---

## Installation

### From PyPI

```bash
pip install aumai-omnipercept
```

### From source (development)

```bash
git clone https://github.com/aumai/aumai-omnipercept.git
cd aumai-omnipercept
pip install -e ".[dev]"
```

### Verify the installation

```bash
omnipercept --version
omnipercept models
```

The second command should list 12 built-in model cards.

---

## Step-by-Step Tutorial

### Step 1 — Understand the core abstraction

Everything starts with a `ModalityInput`. It wraps any data — a string, a file path, a base64 blob — together with its modality and optional metadata.

```python
from aumai_omnipercept import ModalityInput, Modality

# Text input
text = ModalityInput(modality=Modality.TEXT, data="The atmosphere on Mars is 95% CO2")

# Image input with spatial metadata
image = ModalityInput(
    modality=Modality.IMAGE,
    data="/data/mars_surface.jpg",
    width=4096,
    height=3072,
    channels=3,
)

# Audio input with temporal metadata
audio = ModalityInput(
    modality=Modality.AUDIO,
    data="/data/radio_transmission.wav",
    sample_rate=44100,
    duration_seconds=12.5,
)
```

The `data` field is always a string. The library uses it as a deterministic seed for feature extraction. When you swap in a real model backend, `data` becomes the path or bytes you pass to your inference code.

### Step 2 — Extract a feature vector

```python
from aumai_omnipercept import FeatureExtractor

extractor = FeatureExtractor()
fv = extractor.extract(text, target_dim=256)

print(f"Modality:        {fv.modality.value}")       # text
print(f"Dimensions:      {fv.dimensions}")             # 256
print(f"Norm:            {fv.norm:.6f}")               # ~1.0 (unit vector)
print(f"Model used:      {fv.model_name}")             # TextEmbedder
print(f"Extraction time: {fv.extraction_time_ms:.2f} ms")
print(f"First 4 values:  {fv.values[:4]}")
```

The same input always produces the same vector — this determinism makes tests trivial and pipelines reproducible.

### Step 3 — Fuse features from multiple modalities

```python
from aumai_omnipercept import ModalityFusion, FusionStrategy

text_fv  = extractor.extract(text,  target_dim=256)
image_fv = extractor.extract(image, target_dim=256)

fusion = ModalityFusion()
fused = fusion.fuse(
    features=[text_fv, image_fv],
    strategy=FusionStrategy.CROSS_ATTENTION,
    target_dim=256,
)

print(f"Modalities: {[m.value for m in fused.modalities]}")
print(f"Strategy:   {fused.strategy.value}")
print(f"Weights:    {fused.weights}")    # fractional energy per modality
print(f"Time:       {fused.fusion_time_ms:.2f} ms")
```

`weights` shows how much each modality contributed to the fused vector. Use this to debug pipelines where one signal dominates unexpectedly.

### Step 4 — Run an end-to-end perception pipeline

```python
from aumai_omnipercept import PerceptionPipeline, PipelineConfig, TaskType

config = PipelineConfig(
    name="MarsAnalyser",
    modalities=[Modality.TEXT, Modality.IMAGE],
    task=TaskType.CLASSIFICATION,
    fusion_strategy=FusionStrategy.CROSS_ATTENTION,
    feature_dim=256,
    max_predictions=5,
    confidence_threshold=0.05,
)
pipeline = PerceptionPipeline(config)

result = pipeline.process([text, image])

print(f"Task:       {result.task.value}")
print(f"Confidence: {result.confidence:.4f}")
print(f"Time:       {result.processing_time_ms:.2f} ms")
for pred in result.predictions:
    print(f"  {pred.label}: {pred.confidence:.4f}")
```

### Step 5 — Search for similar items

```python
from aumai_omnipercept import SimilaritySearch

index = SimilaritySearch()

corpus = [
    ("mars-article",  "Martian atmosphere composition and pressure"),
    ("venus-article", "Venusian clouds and surface temperature"),
    ("earth-article", "Earth's nitrogen-oxygen atmosphere"),
    ("moon-article",  "Lunar regolith and lack of atmosphere"),
]
for item_id, text_data in corpus:
    fv = extractor.extract(ModalityInput(modality=Modality.TEXT, data=text_data))
    index.add(item_id, fv)

query_fv = extractor.extract(ModalityInput(
    modality=Modality.TEXT,
    data="CO2 atmosphere of terrestrial planets",
))
for item_id, score in index.search(query_fv, top_k=3):
    print(f"  {item_id}: {score:.6f}")
```

### Step 6 — Evaluate with Benchmarker

```python
from aumai_omnipercept import Benchmarker

test_samples = [
    [ModalityInput(modality=Modality.TEXT, data="The price rose sharply this quarter")],
    [ModalityInput(modality=Modality.TEXT, data="Product received rave reviews from customers")],
    [ModalityInput(modality=Modality.TEXT, data="Quarterly earnings beat analyst expectations")],
]
ground_truth = ["class_0", "class_1", "class_0"]

result = Benchmarker().evaluate(pipeline, test_samples, ground_truth, "finance_test")
print(f"Accuracy:  {result.accuracy:.2%}")
print(f"F1 score:  {result.f1_score:.4f}")
print(f"Latency:   {result.avg_latency_ms:.2f} ms/sample")
print(f"Samples:   {result.samples_evaluated}")
```

---

## Five Common Patterns

### Pattern 1 — Text-only classification

```python
from aumai_omnipercept import (
    PerceptionPipeline, PipelineConfig, ModalityInput,
    Modality, TaskType, FusionStrategy,
)

config = PipelineConfig(
    name="text-sentiment",
    modalities=[Modality.TEXT],
    task=TaskType.CLASSIFICATION,
    fusion_strategy=FusionStrategy.LATE,
    feature_dim=128,
    max_predictions=3,
    confidence_threshold=0.1,
)
pipeline = PerceptionPipeline(config)

result = pipeline.process([ModalityInput(modality=Modality.TEXT, data="I love this product!")])
top = result.predictions[0] if result.predictions else None
if top:
    print(f"Top class: {top.label}, confidence: {top.confidence:.4f}")
```

### Pattern 2 — Image + text document understanding

```python
config = PipelineConfig(
    name="document-type-classifier",
    modalities=[Modality.TEXT, Modality.IMAGE],
    task=TaskType.CLASSIFICATION,
    fusion_strategy=FusionStrategy.WEIGHTED_SUM,
    feature_dim=256,
    max_predictions=5,
    confidence_threshold=0.05,
)
pipeline = PerceptionPipeline(config)

inputs = [
    ModalityInput(modality=Modality.TEXT, data="Invoice #12345 from ACME Corp"),
    ModalityInput(modality=Modality.IMAGE, data="invoice_scan.png",
                  width=2480, height=3508, channels=1),
]
result = pipeline.process(inputs)
```

### Pattern 3 — Cross-modal retrieval index

```python
from aumai_omnipercept import FeatureExtractor, SimilaritySearch, ModalityInput, Modality

extractor = FeatureExtractor()
index = SimilaritySearch()

items = [
    ModalityInput(modality=Modality.TEXT,  data="Product manual for Model X"),
    ModalityInput(modality=Modality.IMAGE, data="manual_cover.jpg", width=210, height=297, channels=1),
    ModalityInput(modality=Modality.TEXT,  data="Safety warning for Model X high voltage"),
]
for i, item in enumerate(items):
    index.add(f"item-{i}", extractor.extract(item))

query_fv = extractor.extract(ModalityInput(modality=Modality.TEXT, data="product documentation"))
print(index.search(query_fv, top_k=2))
```

### Pattern 4 — Object detection with bounding boxes

```python
config = PipelineConfig(
    name="scene-detection",
    modalities=[Modality.IMAGE],
    task=TaskType.DETECTION,
    feature_dim=256,
    max_predictions=15,
    confidence_threshold=0.3,
)
pipeline = PerceptionPipeline(config)
result = pipeline.process([
    ModalityInput(modality=Modality.IMAGE, data="warehouse.jpg",
                  width=1920, height=1080, channels=3),
])
for pred in result.predictions:
    if pred.bbox:
        print(f"{pred.label}: {pred.confidence:.3f} @ {pred.bbox}")
```

### Pattern 5 — Plug in real model inference

To use real ML models, subclass `FeatureExtractor` and override `extract()`:

```python
import time
from aumai_omnipercept import FeatureExtractor, FeatureVector, ModalityInput, Modality

class RealTextExtractor(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract(self, inp: ModalityInput, target_dim: int = 256) -> FeatureVector:
        if inp.modality != Modality.TEXT:
            return super().extract(inp, target_dim)
        start = time.monotonic()
        embedding: list[float] = self._model.encode(
            inp.data, normalize_embeddings=True
        ).tolist()
        # Resize to target_dim
        if len(embedding) > target_dim:
            embedding = embedding[:target_dim]
        else:
            embedding = embedding + [0.0] * (target_dim - len(embedding))
        return FeatureVector(
            modality=inp.modality,
            dimensions=target_dim,
            values=embedding,
            model_name="all-MiniLM-L6-v2",
            extraction_time_ms=round((time.monotonic() - start) * 1000, 2),
        )
```

---

## FAQ

**Q: Why does the same text always produce the same feature vector?**

The built-in `FeatureExtractor` hashes `"{modality}:{data}"` with SHA-256 and expands that seed deterministically into floats. There is no randomness. This makes tests stable and pipelines reproducible without a GPU.

**Q: How do I use a real ML model?**

See Pattern 5 above. Subclass `FeatureExtractor`, override `extract()`, and inject your instance into the pipeline via `pipeline._extractor = your_extractor`.

**Q: What does `confidence_threshold` control?**

Predictions whose confidence falls below this value are excluded from `PerceptionResult.predictions`. Set it to `0.0` to receive all predictions; raise it to filter to high-confidence ones only.

**Q: How does `SimilaritySearch` scale?**

It uses a linear scan — O(n) per query. This works well up to tens of thousands of items. For production scale, extract `FeatureVector.values` and load them into FAISS, Weaviate, Milvus, or pgvector.

**Q: What is the difference between EARLY and LATE fusion?**

`EARLY` concatenates all vectors then stride-samples to `target_dim`. It preserves cross-modal relationships but compresses information. `LATE` averages independently after zero-padding. It is more robust when modalities have different information densities.

**Q: Can I serialise a `FeatureVector` to disk?**

Yes. `FeatureVector` is Pydantic v2, so:

```python
# Save
json_str = fv.model_dump_json()

# Load
from aumai_omnipercept import FeatureVector
fv_loaded = FeatureVector.model_validate_json(json_str)
```

**Q: Why do `FusedRepresentation.weights` not sum to exactly 1.0?**

Each weight is rounded to 4 decimal places. The rounding error is at most `0.0001 * num_modalities`. The underlying floating-point values sum to exactly 1.0.

---

## Next Steps

- [API Reference](api-reference.md) — complete signatures for every public class and function
- [examples/quickstart.py](../examples/quickstart.py) — runnable demo covering all five patterns
- [CONTRIBUTING.md](../CONTRIBUTING.md) — how to add modalities, fusion strategies, or model cards
