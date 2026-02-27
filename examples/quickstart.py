"""Quickstart examples for aumai-omnipercept.

Demonstrates five core patterns:
  1. Single-modality text feature extraction
  2. Multi-modal fusion (text + image + audio)
  3. End-to-end perception pipeline (classification)
  4. Cosine similarity search index
  5. Benchmarking a pipeline

Run directly to verify your installation:

    python examples/quickstart.py
"""

from __future__ import annotations

from aumai_omnipercept import (
    Benchmarker,
    FeatureExtractor,
    FusionStrategy,
    Modality,
    ModalityFusion,
    ModalityInput,
    ModelRegistry,
    PerceptionPipeline,
    PipelineConfig,
    SimilaritySearch,
    TaskType,
)


# ---------------------------------------------------------------------------
# Demo 1: Feature extraction
# ---------------------------------------------------------------------------

def demo_feature_extraction() -> None:
    """Extract L2-normalised feature vectors from text, image, and audio inputs.

    The built-in extractor uses SHA-256-seeded deterministic vectors —
    no GPU or model download required. Same input always yields the same vector.
    """
    print("=" * 60)
    print("Demo 1: Feature Extraction")
    print("=" * 60)

    extractor = FeatureExtractor()

    # Text input
    text_input = ModalityInput(
        modality=Modality.TEXT,
        data="The mission is to bring clean water to every household in India",
    )
    text_fv = extractor.extract(text_input, target_dim=128)
    print(f"\nText feature:")
    print(f"  Model:      {text_fv.model_name}")
    print(f"  Dimensions: {text_fv.dimensions}")
    print(f"  Norm:       {text_fv.norm:.6f}  (should be ~1.0 — unit vector)")
    print(f"  Time:       {text_fv.extraction_time_ms:.2f} ms")
    print(f"  First 6:    {[round(v, 4) for v in text_fv.values[:6]]}")

    # Image input with spatial metadata
    image_input = ModalityInput(
        modality=Modality.IMAGE,
        data="/data/satellite_image.jpg",
        width=3840,
        height=2160,
        channels=3,
    )
    image_fv = extractor.extract(image_input, target_dim=256)
    print(f"\nImage feature:")
    print(f"  Model:      {image_fv.model_name}")
    print(f"  Dimensions: {image_fv.dimensions}")
    print(f"  Norm:       {image_fv.norm:.6f}")

    # Audio input with temporal metadata
    audio_input = ModalityInput(
        modality=Modality.AUDIO,
        data="/data/pump_sound.wav",
        sample_rate=44100,
        duration_seconds=8.0,
    )
    audio_fv = extractor.extract(audio_input, target_dim=128)
    print(f"\nAudio feature:")
    print(f"  Model:      {audio_fv.model_name}")
    print(f"  Dimensions: {audio_fv.dimensions}")
    print(f"  Norm:       {audio_fv.norm:.6f}")


# ---------------------------------------------------------------------------
# Demo 2: Multi-modal fusion
# ---------------------------------------------------------------------------

def demo_fusion() -> None:
    """Fuse features from three modalities using all four fusion strategies.

    The 'weights' field shows each modality's fractional energy contribution.
    Use this to understand which signal dominated the fused representation.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Multi-modal Fusion")
    print("=" * 60)

    extractor = FeatureExtractor()
    fusion = ModalityFusion()

    inputs = [
        ModalityInput(modality=Modality.TEXT,    data="Weather alert: heavy rainfall expected"),
        ModalityInput(modality=Modality.IMAGE,   data="radar_map.png", width=800, height=600, channels=3),
        ModalityInput(modality=Modality.TABULAR, data="rainfall_mm,pressure_hpa\n85.2,998.1"),
    ]
    features = [extractor.extract(inp, target_dim=128) for inp in inputs]

    print(f"\nExtracted {len(features)} feature vectors:")
    for fv in features:
        print(f"  {fv.modality.value:<12s} dim={fv.dimensions} norm={fv.norm:.4f}")

    print("\nFusion strategies comparison:")
    for strategy in FusionStrategy:
        fused = fusion.fuse(features, strategy, target_dim=128)
        weights_str = ", ".join(f"{k}={v:.3f}" for k, v in fused.weights.items())
        print(f"  {strategy.value:<18s} time={fused.fusion_time_ms:.3f}ms  weights=[{weights_str}]")


# ---------------------------------------------------------------------------
# Demo 3: End-to-end perception pipeline
# ---------------------------------------------------------------------------

def demo_perception_pipeline() -> None:
    """Run a full classification pipeline over text + image inputs.

    PipelineConfig declares which modalities and task to use.
    PerceptionPipeline.process() handles extraction, fusion, and prediction.
    """
    print("\n" + "=" * 60)
    print("Demo 3: End-to-End Perception Pipeline")
    print("=" * 60)

    config = PipelineConfig(
        name="DocumentTypeClassifier",
        modalities=[Modality.TEXT, Modality.IMAGE],
        task=TaskType.CLASSIFICATION,
        fusion_strategy=FusionStrategy.CROSS_ATTENTION,
        feature_dim=256,
        max_predictions=5,
        confidence_threshold=0.05,
    )
    pipeline = PerceptionPipeline(config)

    test_cases = [
        (
            "Invoice #A-4471 from BuildRight Construction — total $28,450",
            ModalityInput(modality=Modality.IMAGE, data="invoice_scan.png", width=2480, height=3508, channels=1),
        ),
        (
            "Dear Mr. Patel, We are pleased to offer you the position of Senior Engineer",
            ModalityInput(modality=Modality.IMAGE, data="offer_letter.png", width=2480, height=3508, channels=1),
        ),
    ]

    for text_data, image_input in test_cases:
        text_input = ModalityInput(modality=Modality.TEXT, data=text_data)
        result = pipeline.process([text_input, image_input])

        print(f"\nInput: '{text_data[:55]}...'")
        print(f"  Task:        {result.task.value}")
        print(f"  Modalities:  {[m.value for m in result.modalities_used]}")
        print(f"  Confidence:  {result.confidence:.4f}")
        print(f"  Time:        {result.processing_time_ms:.2f} ms")
        print(f"  Predictions:")
        for pred in result.predictions[:3]:
            print(f"    {pred.label}: {pred.confidence:.4f}")


# ---------------------------------------------------------------------------
# Demo 4: Similarity search index
# ---------------------------------------------------------------------------

def demo_similarity_search() -> None:
    """Build an in-memory search index and query it.

    All indexed items use the same vector space — cross-modality search
    works naturally because cosine similarity is modality-agnostic.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Similarity Search")
    print("=" * 60)

    extractor = FeatureExtractor()
    index = SimilaritySearch()

    # Index a mixed corpus of text and image items
    corpus = [
        ("article-groundwater",  Modality.TEXT,  "Groundwater depletion in Rajasthan borewells 2023"),
        ("article-rainfall",     Modality.TEXT,  "IMD monsoon rainfall forecast for Vidarbha region"),
        ("article-treatment",    Modality.TEXT,  "Iron removal plant design for rural borewell water"),
        ("article-mission",      Modality.TEXT,  "Jal Jeevan Mission household tap connection targets"),
        ("article-quality",      Modality.TEXT,  "BIS 10500 drinking water quality standards fluoride"),
        ("satellite-rajasthan",  Modality.IMAGE, "satellite_groundwater_map.tif"),
        ("satellite-maharashtra",Modality.IMAGE, "satellite_rainfall_index.tif"),
    ]

    print("\nIndexing corpus:")
    for item_id, modality, data in corpus:
        if modality == Modality.IMAGE:
            inp = ModalityInput(modality=modality, data=data, width=512, height=512, channels=1)
        else:
            inp = ModalityInput(modality=modality, data=data)
        fv = extractor.extract(inp)
        index.add(item_id, fv)
        print(f"  Added: {item_id}")

    # Run several queries
    queries = [
        "drinking water quality in rural India",
        "monsoon seasonal rainfall data",
        "fluoride contamination treatment methods",
    ]

    print("\nSearch results:")
    for query_text in queries:
        query_fv = extractor.extract(ModalityInput(modality=Modality.TEXT, data=query_text))
        results = index.search(query_fv, top_k=3)
        print(f"\n  Query: '{query_text}'")
        for item_id, score in results:
            print(f"    {item_id:<32s} sim={score:.6f}")


# ---------------------------------------------------------------------------
# Demo 5: Pipeline benchmarking
# ---------------------------------------------------------------------------

def demo_benchmarking() -> None:
    """Evaluate a pipeline against labelled test data.

    Benchmarker.evaluate() computes top-1 accuracy plus macro
    precision, recall, and F1 — the same metrics used for
    multi-class classification evaluation.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Pipeline Benchmarking")
    print("=" * 60)

    config = PipelineConfig(
        name="AudioEventClassifier",
        modalities=[Modality.AUDIO],
        task=TaskType.CLASSIFICATION,
        fusion_strategy=FusionStrategy.LATE,
        feature_dim=128,
        max_predictions=5,
        confidence_threshold=0.0,  # show all predictions for benchmark
    )
    pipeline = PerceptionPipeline(config)

    test_inputs = [
        [ModalityInput(modality=Modality.AUDIO, data="pump_running_normal.wav",   sample_rate=16000, duration_seconds=5.0)],
        [ModalityInput(modality=Modality.AUDIO, data="pump_vibration_fault.wav",  sample_rate=16000, duration_seconds=5.0)],
        [ModalityInput(modality=Modality.AUDIO, data="motor_hum_baseline.wav",    sample_rate=22050, duration_seconds=3.0)],
        [ModalityInput(modality=Modality.AUDIO, data="pipe_leak_drip.wav",        sample_rate=44100, duration_seconds=7.0)],
        [ModalityInput(modality=Modality.AUDIO, data="valve_open_close.wav",      sample_rate=16000, duration_seconds=2.5)],
        [ModalityInput(modality=Modality.AUDIO, data="pump_running_normal_2.wav", sample_rate=16000, duration_seconds=5.0)],
    ]
    ground_truth = ["class_0", "class_1", "class_2", "class_1", "class_0", "class_0"]

    benchmarker = Benchmarker()
    result = benchmarker.evaluate(
        pipeline=pipeline,
        test_inputs=test_inputs,
        ground_truth=ground_truth,
        dataset_name="pump_audio_v1",
    )

    print(f"\nBenchmark: {result.pipeline_name} on '{result.dataset}'")
    print(f"  Task:              {result.task.value}")
    print(f"  Samples evaluated: {result.samples_evaluated}")
    print(f"  Accuracy:          {result.accuracy:.4f}  ({result.accuracy:.1%})")
    print(f"  Precision (macro): {result.precision:.4f}")
    print(f"  Recall (macro):    {result.recall:.4f}")
    print(f"  F1 Score (macro):  {result.f1_score:.4f}")
    print(f"  Avg latency:       {result.avg_latency_ms:.2f} ms/sample")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all five quickstart demos."""
    print("\naumai-omnipercept Quickstart")
    print("Multimodal perception framework — five core patterns\n")

    demo_feature_extraction()
    demo_fusion()
    demo_perception_pipeline()
    demo_similarity_search()
    demo_benchmarking()

    print("\n" + "=" * 60)
    print("All demos completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
