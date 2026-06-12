"""
Run OK-VQA evaluation for OV-VQA system.

This script:
1. loads configs/default.yaml and configs/okvqa.yaml
2. builds OK-VQA dataset
3. builds OV-VQA pipeline
4. runs inference sample by sample
5. computes OK-VQA VQA-style accuracy
6. saves:
   - config.yaml
   - command.txt
   - run.log
   - predictions.jsonl
   - intermediate.jsonl
   - metrics.json

Example:
    python scripts/run_okvqa_eval.py \
        --config configs/okvqa.yaml \
        --limit 50 \
        --output_dir outputs/okvqa/smoke_full

For extra config override:
    python scripts/run_okvqa_eval.py \
        --config configs/okvqa.yaml \
        --override detector.conf_threshold=0.3 \
        --override retrieval.top_k=20
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# Make project root importable when running:
# python scripts/run_okvqa_eval.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.build_dataset import build_dataset, dataset_summary
from src.evaluation.okvqa_metric import build_okvqa_metric
from src.pipelines.vqa_pipeline import build_vqa_pipeline
from src.utils.config import load_config, parse_cli_overrides, save_config
from src.utils.io import (
    initialize_output_files,
    save_intermediate,
    save_metrics,
    save_prediction,
)
from src.utils.logger import (
    get_log_level,
    get_logger,
    log_config_summary,
    log_final_metrics,
    log_sample_result,
    log_sample_start,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run OK-VQA evaluation for OV-VQA system."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/okvqa.yaml",
        help="Path to experiment config.",
    )

    parser.add_argument(
        "--default_config",
        type=str,
        default="configs/default.yaml",
        help="Path to default config.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override experiment.output_dir.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override data.limit for smoke tests.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Override experiment.overwrite to true.",
    )

    parser.add_argument(
        "--catch_errors",
        action="store_true",
        help="Catch per-sample errors and continue running.",
    )

    parser.add_argument(
        "--skip_llm_check",
        action="store_true",
        help="Skip local Ollama availability check.",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Logging level: debug, info, warning, error.",
    )

    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override config by dot key. "
            "Example: --override detector.conf_threshold=0.3"
        ),
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build CLI config overrides.

    Args:
        args: Parsed arguments.

    Returns:
        Dot-key override dictionary.
    """
    overrides = parse_cli_overrides(args.override)

    if args.output_dir is not None:
        overrides["experiment.output_dir"] = args.output_dir

    if args.limit is not None:
        overrides["data.limit"] = args.limit

    if args.overwrite:
        overrides["experiment.overwrite"] = True

    return overrides


def check_llm_available(pipeline: Any, logger: Any, skip_check: bool = False) -> None:
    """
    Check local Ollama server before running the experiment.

    Args:
        pipeline: OVVQAPipeline.
        logger: Logger.
        skip_check: Whether to skip the check.

    Raises:
        RuntimeError: If Ollama is unavailable.
    """
    if skip_check:
        logger.warning("Skipping Ollama availability check.")
        return

    llm = getattr(pipeline, "llm", None)

    if llm is None or not hasattr(llm, "check_available"):
        logger.warning("LLM availability check is not supported by this wrapper.")
        return

    available = llm.check_available()

    if not available:
        raise RuntimeError(
            "Local Ollama server is not available. "
            "Please start it in AutoDL terminal, for example: ollama serve"
        )

    logger.info("Local Ollama server is available.")

    if hasattr(llm, "list_models"):
        try:
            models = llm.list_models()
            logger.info("Available Ollama models: %s", models)
        except Exception as e:
            logger.warning("Could not list Ollama models: %s", e)


def save_one_result(
    result: Any,
    sample: Dict[str, Any],
    score: float,
    paths: Dict[str, Path],
) -> None:
    """
    Save one pipeline result to predictions.jsonl and intermediate.jsonl.

    Args:
        result: PipelineResult.
        sample: Unified VQA sample.
        score: Per-sample OK-VQA score.
        paths: Output paths.
    """
    result_dict = result.to_dict()

    save_prediction(
        path=paths["predictions"],
        question_id=sample.get("question_id"),
        image_id=sample.get("image_id"),
        question=sample.get("question", ""),
        pred_answer=result.pred_answer,
        clean_answer=result.clean_answer,
        gt_answers=sample.get("answers", []),
        score=score,
        extra={
            "error": result.error,
            "latency": result.latency,
        },
    )

    save_intermediate(
        path=paths["intermediate"],
        sample=sample,
        concepts=result.concepts,
        expanded_concepts=result.expanded_concepts,
        selected_prompts=result.selected_prompts,
        detections=result.detections,
        retrieved_passages=result.retrieved_passages,
        reranked_passages=result.reranked_passages,
        visual_evidence=result.visual_evidence,
        reasoning_input=result.reasoning_input,
        raw_reasoning_output=result.raw_reasoning_output,
        pred_answer=result.pred_answer,
        clean_answer=result.clean_answer,
        score=score,
        extra={
            "answer_candidates": result.answer_candidates,
            "knowledge_evidence": result.knowledge_evidence,
            "final_prompt": result.final_prompt,
            "raw_concept_generation_output": result.raw_concept_generation_output,
            "raw_prompt_selection_output": result.raw_prompt_selection_output,
            "raw_candidate_generation_output": result.raw_candidate_generation_output,
            "candidate_generation_prompt": result.candidate_generation_prompt,
            "latency": result.latency,
            "error": result.error,
        },
    )


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()
    cli_overrides = build_cli_overrides(args)

    cfg = load_config(
        config_path=args.config,
        default_path=args.default_config,
        cli_overrides=cli_overrides,
        validate=True,
    )

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    paths = initialize_output_files(cfg)
    save_config(cfg, paths["config"])

    logger = get_logger(
        name="okvqa_eval",
        log_file=paths["log"],
        level=get_log_level(args.log_level),
    )

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Config path: %s", args.config)
    logger.info("Default config path: %s", args.default_config)

    log_config_summary(logger, cfg)

    logger.info("Building dataset...")
    dataset = build_dataset(cfg)
    ds_summary = dataset_summary(dataset)
    logger.info("Dataset summary: %s", ds_summary)

    logger.info("Building pipeline...")
    pipeline = build_vqa_pipeline(cfg)

    check_llm_available(
        pipeline=pipeline,
        logger=logger,
        skip_check=args.skip_llm_check,
    )

    metric = build_okvqa_metric(cfg)

    total = len(dataset)
    logger.info("Starting OK-VQA evaluation on %d samples.", total)

    start_time = time.time()
    num_errors = 0

    for index, sample in enumerate(dataset, start=1):
        log_sample_start(
            logger=logger,
            index=index,
            total=total,
            question_id=sample.get("question_id"),
            image_id=sample.get("image_id"),
            question=sample.get("question", ""),
        )

        result = pipeline.run(
            sample=sample,
            catch_errors=args.catch_errors,
        )

        if result.error:
            num_errors += 1
            logger.error("Sample error: %s", result.error)

        score = metric.add(
            prediction=result.clean_answer,
            gt_answers=sample.get("answers", []),
            question_id=sample.get("question_id"),
            image_id=sample.get("image_id"),
            question=sample.get("question", ""),
        )

        save_one_result(
            result=result,
            sample=sample,
            score=score,
            paths=paths,
        )

        log_sample_result(
            logger=logger,
            pred_answer=result.pred_answer,
            clean_answer=result.clean_answer,
            gt_answers=sample.get("answers", []),
            score=score,
        )

        current_metrics = metric.compute()
        logger.info(
            "Running accuracy: %.6f (%d samples)",
            current_metrics["accuracy"],
            current_metrics["num_samples"],
        )

    metrics = metric.compute()

    elapsed = time.time() - start_time
    metrics.update(
        {
            "dataset_summary": ds_summary,
            "num_errors": num_errors,
            "elapsed_seconds": elapsed,
            "avg_seconds_per_sample": elapsed / total if total > 0 else 0.0,
        }
    )

    save_metrics(
        path=paths["metrics"],
        metrics=metrics,
        cfg=cfg,
    )

    log_final_metrics(logger, metrics)

    logger.info("Saved predictions to: %s", paths["predictions"])
    logger.info("Saved intermediate results to: %s", paths["intermediate"])
    logger.info("Saved metrics to: %s", paths["metrics"])
    logger.info("Done.")


if __name__ == "__main__":
    main()