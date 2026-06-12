from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")

    return data


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def set_by_dot_key(config: Dict[str, Any], dot_key: str, value: Any) -> None:
    parts = dot_key.split(".")
    current = config

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def parse_dot_override(item: str) -> Tuple[str, Any]:
    if "=" not in item:
        raise ValueError(
            f"Invalid override: {item}. Expected format: key.subkey=value"
        )

    key, raw_value = item.split("=", 1)
    key = key.strip()

    if not key:
        raise ValueError(f"Invalid override key: {item}")

    return key, parse_override_value(raw_value.strip())


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def discover_ablation_files(
    ablation_dir: Path,
    selected_names: Optional[Sequence[str]] = None,
) -> List[Path]:
    if selected_names:
        files = []

        for name in selected_names:
            p = Path(name)

            if p.suffix in {".yaml", ".yml"}:
                candidate = p if p.is_absolute() else ablation_dir / p.name
            else:
                candidate = ablation_dir / f"{name}.yaml"

            if not candidate.exists():
                alt = ablation_dir / f"{name}.yml"
                if alt.exists():
                    candidate = alt

            if not candidate.exists():
                raise FileNotFoundError(f"Ablation config not found: {candidate}")

            files.append(candidate)

        return files

    files = sorted(list(ablation_dir.glob("*.yaml")) + list(ablation_dir.glob("*.yml")))

    if not files:
        raise FileNotFoundError(f"No ablation YAML files found in: {ablation_dir}")

    return files


def read_metrics(output_dir: Path) -> Dict[str, Any]:
    metrics_path = output_dir / "metrics.json"

    if not metrics_path.exists():
        return {
            "metrics_found": False,
            "metrics_path": str(metrics_path),
        }

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    metrics["metrics_found"] = True
    metrics["metrics_path"] = str(metrics_path)

    return metrics


def write_summary(
    rows: List[Dict[str, Any]],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "ablation_summary.json"
    csv_path = output_root / "ablation_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "name",
        "status",
        "return_code",
        "accuracy",
        "okvqa_accuracy",
        "correct_sum",
        "num_samples",
        "elapsed_seconds",
        "avg_seconds_per_sample",
        "output_dir",
        "config_path",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"\nSaved summary JSON to: {json_path}")
    print(f"Saved summary CSV to:  {csv_path}")


def build_command(
    python_bin: str,
    eval_script: Path,
    merged_config_path: Path,
    output_dir: Path,
    limit: Optional[int],
    shuffle: bool,
    overwrite: bool,
    extra_eval_args: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd = [
        python_bin,
        str(eval_script),
        "--config",
        str(merged_config_path),
        "--output_dir",
        str(output_dir),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if shuffle:
        cmd.append("--shuffle")

    if overwrite:
        cmd.append("--overwrite")

    if extra_eval_args:
        cmd.extend(extra_eval_args)

    return cmd


def run_one_ablation(
    *,
    name: str,
    base_config: Dict[str, Any],
    ablation_config_path: Path,
    eval_script: Path,
    output_root: Path,
    merged_config_dir: Path,
    python_bin: str,
    limit: Optional[int],
    shuffle: bool,
    overwrite: bool,
    offline: bool,
    respect_ablation_output_dir: bool,
    dot_overrides: Sequence[str],
    dry_run: bool,
    extra_eval_args: Optional[Sequence[str]],
    project_root: Path,
) -> Dict[str, Any]:
    print("\n" + "=" * 100)
    print(f"Running ablation: {name}")
    print(f"Ablation config: {ablation_config_path}")
    print("=" * 100)

    ablation_config = load_yaml(ablation_config_path)
    merged = deep_merge(base_config, ablation_config)

    for item in dot_overrides:
        key, value = parse_dot_override(item)
        set_by_dot_key(merged, key, value)

    if not respect_ablation_output_dir:
        run_output_dir = output_root / name
        set_by_dot_key(merged, "output.output_dir", str(run_output_dir))
    else:
        configured_output = (
            merged.get("output", {}).get("output_dir")
            if isinstance(merged.get("output"), dict)
            else None
        )
        run_output_dir = Path(configured_output) if configured_output else output_root / name
        if not run_output_dir.is_absolute():
            run_output_dir = project_root / run_output_dir

    experiment_name = merged.get("experiment", {}).get("name", name)
    if isinstance(merged.get("experiment"), dict):
        merged["experiment"]["name"] = experiment_name

    merged_config_path = merged_config_dir / f"{name}.yaml"
    save_yaml(merged, merged_config_path)

    cmd = build_command(
        python_bin=python_bin,
        eval_script=eval_script,
        merged_config_path=merged_config_path,
        output_dir=run_output_dir,
        limit=limit,
        shuffle=shuffle,
        overwrite=overwrite,
        extra_eval_args=extra_eval_args,
    )

    print("Merged config:", merged_config_path)
    print("Output dir:   ", run_output_dir)
    print("Command:")
    print(" ".join(cmd))

    row: Dict[str, Any] = {
        "name": name,
        "status": "dry_run" if dry_run else "pending",
        "return_code": None,
        "output_dir": str(run_output_dir),
        "config_path": str(merged_config_path),
    }

    if dry_run:
        return row

    env = os.environ.copy()

    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"

    start_time = time.time()

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        text=True,
    )

    wall_time = time.time() - start_time

    row["return_code"] = proc.returncode
    row["wall_time_seconds"] = wall_time

    if proc.returncode == 0:
        row["status"] = "success"
        metrics = read_metrics(run_output_dir)

        row.update(
            {
                "accuracy": metrics.get("accuracy"),
                "okvqa_accuracy": metrics.get("okvqa_accuracy"),
                "correct_sum": metrics.get("correct_sum"),
                "num_samples": metrics.get("num_samples"),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
                "avg_seconds_per_sample": metrics.get("avg_seconds_per_sample"),
                "metrics_found": metrics.get("metrics_found"),
                "metrics_path": metrics.get("metrics_path"),
            }
        )
    else:
        row["status"] = "failed"

    print(f"Ablation finished: {name}")
    print(f"Status: {row['status']}")
    print(f"Return code: {row['return_code']}")

    if row.get("accuracy") is not None:
        print(f"Accuracy: {row.get('accuracy')}")

    return row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OV-VQA ablation experiments by merging base and ablation YAML configs."
    )

    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/okvqa.yaml",
        help="Base experiment config.",
    )

    parser.add_argument(
        "--ablation_dir",
        type=str,
        default="configs/ablations",
        help="Directory containing ablation YAML files.",
    )

    parser.add_argument(
        "--ablations",
        nargs="*",
        default=None,
        help=(
            "Ablation names or YAML files to run. "
            "Example: --ablations no_retrieval no_reranker"
        ),
    )

    parser.add_argument(
        "--eval_script",
        type=str,
        default="scripts/run_okvqa_eval.py",
        help="Evaluation script to execute.",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/okvqa/ablations",
        help="Root directory for ablation outputs.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to evaluate. Omit for full split.",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset before applying limit.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to the evaluation script.",
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Set HF offline environment variables. Enabled by default.",
    )

    parser.add_argument(
        "--no_offline",
        action="store_false",
        dest="offline",
        help="Disable HF offline environment variables.",
    )

    parser.add_argument(
        "--respect_ablation_output_dir",
        action="store_true",
        help=(
            "Use output.output_dir from each ablation YAML instead of "
            "forcing output_root/ablation_name."
        ),
    )

    parser.add_argument(
        "--set",
        dest="dot_overrides",
        action="append",
        default=[],
        help=(
            "Extra dot-key override applied after config merge. "
            "Example: --set detector.max_prompts=5"
        ),
    )

    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and write merged configs without running experiments.",
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running remaining ablations if one fails.",
    )

    parser.add_argument(
        "--extra_eval_args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Additional arguments passed directly to the evaluation script.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]

    base_config_path = Path(args.base_config)
    ablation_dir = Path(args.ablation_dir)
    eval_script = Path(args.eval_script)
    output_root = Path(args.output_root)

    if not base_config_path.is_absolute():
        base_config_path = project_root / base_config_path

    if not ablation_dir.is_absolute():
        ablation_dir = project_root / ablation_dir

    if not eval_script.is_absolute():
        eval_script = project_root / eval_script

    if not output_root.is_absolute():
        output_root = project_root / output_root

    if not eval_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {eval_script}")

    base_config = load_yaml(base_config_path)
    ablation_files = discover_ablation_files(ablation_dir, args.ablations)

    merged_config_dir = output_root / "merged_configs"
    output_root.mkdir(parents=True, exist_ok=True)
    merged_config_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:     ", project_root)
    print("Base config:      ", base_config_path)
    print("Ablation dir:     ", ablation_dir)
    print("Eval script:      ", eval_script)
    print("Output root:      ", output_root)
    print("Num ablations:    ", len(ablation_files))
    print("Limit:            ", args.limit)
    print("Shuffle:          ", args.shuffle)
    print("Overwrite:        ", args.overwrite)
    print("Offline:          ", args.offline)

    rows: List[Dict[str, Any]] = []

    for ablation_path in ablation_files:
        name = safe_stem(ablation_path)

        try:
            row = run_one_ablation(
                name=name,
                base_config=base_config,
                ablation_config_path=ablation_path,
                eval_script=eval_script,
                output_root=output_root,
                merged_config_dir=merged_config_dir,
                python_bin=args.python,
                limit=args.limit,
                shuffle=args.shuffle,
                overwrite=args.overwrite,
                offline=args.offline,
                respect_ablation_output_dir=args.respect_ablation_output_dir,
                dot_overrides=args.dot_overrides,
                dry_run=args.dry_run,
                extra_eval_args=args.extra_eval_args,
                project_root=project_root,
            )
            rows.append(row)

            if row.get("status") == "failed" and not args.continue_on_error:
                write_summary(rows, output_root)
                raise RuntimeError(f"Ablation failed: {name}")

        except Exception as exc:
            row = {
                "name": name,
                "status": "error",
                "return_code": None,
                "error": str(exc),
                "output_dir": str(output_root / name),
                "config_path": str(merged_config_dir / f"{name}.yaml"),
            }
            rows.append(row)

            print(f"Error while running ablation {name}: {exc}")

            if not args.continue_on_error:
                write_summary(rows, output_root)
                raise

    write_summary(rows, output_root)

    print("\nAblation results:")
    print("-" * 100)
    for row in rows:
        name = row.get("name")
        status = row.get("status")
        acc = row.get("accuracy")
        n = row.get("num_samples")
        print(f"{name:35s} | {status:10s} | acc={acc} | n={n}")
    print("-" * 100)


if __name__ == "__main__":
    main()