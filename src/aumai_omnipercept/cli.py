"""CLI entry point for aumai-omnipercept."""

from __future__ import annotations

import json

import click

from aumai_omnipercept.core import (
    Benchmarker,
    FeatureExtractor,
    ModalityFusion,
    ModelRegistry,
    PerceptionPipeline,
    SimilaritySearch,
)
from aumai_omnipercept.models import (
    FusionStrategy,
    Modality,
    ModalityInput,
    PipelineConfig,
    TaskType,
)


@click.group()
@click.version_option()
def cli() -> None:
    """AumAI OmniPercept - Multimodal perception framework."""


@cli.command()
@click.option("--modality", type=click.Choice([m.value for m in Modality]), help="Filter by modality")
@click.option("--task", type=click.Choice([t.value for t in TaskType]), help="Filter by task")
def models(modality: str | None, task: str | None) -> None:
    """List available perception models."""
    registry = ModelRegistry()

    if modality:
        results = registry.by_modality(Modality(modality))
    elif task:
        results = registry.by_task(TaskType(task))
    else:
        results = registry.all_models()

    click.echo(f"\n{len(results)} model(s) available:\n")
    for m in results:
        click.echo(f"  {m.model_id}")
        click.echo(f"    {m.name} | {m.modality.value} | {m.task.value}")
        click.echo(f"    Dim: {m.feature_dim} | Params: {m.parameters_millions}M")
        click.echo(f"    {m.description}")
        click.echo()


@cli.command()
@click.option("--data", required=True, help="Input data (text or file path)")
@click.option("--modality", required=True, type=click.Choice([m.value for m in Modality]), help="Input modality")
@click.option("--dim", default=256, type=int, help="Feature dimension")
def extract(data: str, modality: str, dim: int) -> None:
    """Extract feature vector from a single input."""
    inp = ModalityInput(modality=Modality(modality), data=data)
    extractor = FeatureExtractor()
    fv = extractor.extract(inp, target_dim=dim)

    click.echo(f"\nFeature Extraction: {modality}")
    click.echo(f"  Model: {fv.model_name}")
    click.echo(f"  Dimensions: {fv.dimensions}")
    click.echo(f"  Norm: {fv.norm:.6f}")
    click.echo(f"  Time: {fv.extraction_time_ms:.2f} ms")
    click.echo(f"  First 8 values: {[round(v, 4) for v in fv.values[:8]]}")


@cli.command()
@click.option("--input", "input_file", required=True, type=click.Path(exists=True), help="Inputs JSON file")
@click.option("--strategy", type=click.Choice([s.value for s in FusionStrategy]), default="late", help="Fusion strategy")
@click.option("--dim", default=256, type=int, help="Feature dimension")
def fuse(input_file: str, strategy: str, dim: int) -> None:
    """Fuse features from multiple modality inputs."""
    with open(input_file) as f:
        data = json.load(f)

    extractor = FeatureExtractor()
    fusion = ModalityFusion()

    features = []
    for item in data:
        inp = ModalityInput.model_validate(item)
        fv = extractor.extract(inp, target_dim=dim)
        features.append(fv)
        click.echo(f"  Extracted {inp.modality.value}: dim={fv.dimensions}, norm={fv.norm:.4f}")

    result = fusion.fuse(features, FusionStrategy(strategy), target_dim=dim)
    click.echo(f"\nFusion ({strategy}):")
    click.echo(f"  Modalities: {[m.value for m in result.modalities]}")
    click.echo(f"  Dimensions: {result.dimensions}")
    click.echo(f"  Weights: {result.weights}")
    click.echo(f"  Time: {result.fusion_time_ms:.2f} ms")


@cli.command()
@click.option("--config", "config_file", required=True, type=click.Path(exists=True), help="Pipeline config JSON")
@click.option("--input", "input_file", required=True, type=click.Path(exists=True), help="Inputs JSON file")
def perceive(config_file: str, input_file: str) -> None:
    """Run full perception pipeline on multimodal inputs."""
    with open(config_file) as f:
        config_data = json.load(f)
    with open(input_file) as f:
        input_data = json.load(f)

    config = PipelineConfig.model_validate(config_data)
    inputs = [ModalityInput.model_validate(item) for item in input_data]

    pipeline = PerceptionPipeline(config)
    result = pipeline.process(inputs)

    click.echo(f"\nPerception Result: {config.name}")
    click.echo(f"  Task: {result.task.value}")
    click.echo(f"  Modalities: {[m.value for m in result.modalities_used]}")
    click.echo(f"  Confidence: {result.confidence:.4f}")
    click.echo(f"  Time: {result.processing_time_ms:.2f} ms")
    click.echo(f"\n  Predictions ({len(result.predictions)}):")
    for pred in result.predictions:
        bbox_str = f" bbox={pred.bbox}" if pred.bbox else ""
        click.echo(f"    {pred.label}: {pred.confidence:.4f}{bbox_str}")


@cli.command()
@click.option("--query", required=True, help="Query text")
@click.option("--items", required=True, type=click.Path(exists=True), help="Items JSON file (list of {id, data, modality})")
@click.option("--top-k", default=5, type=int, help="Number of results")
def search(query: str, items: str, top_k: int) -> None:
    """Search for similar items using feature similarity."""
    with open(items) as f:
        data = json.load(f)

    extractor = FeatureExtractor()
    index = SimilaritySearch()

    for item in data:
        inp = ModalityInput(modality=Modality(item.get("modality", "text")), data=item["data"])
        fv = extractor.extract(inp)
        index.add(item["id"], fv)

    query_inp = ModalityInput(modality=Modality.TEXT, data=query)
    query_fv = extractor.extract(query_inp)
    results = index.search(query_fv, top_k=top_k)

    click.echo(f"\nSearch results for: '{query}'\n")
    for item_id, score in results:
        click.echo(f"  {item_id}: similarity={score:.6f}")


main = cli

if __name__ == "__main__":
    cli()
