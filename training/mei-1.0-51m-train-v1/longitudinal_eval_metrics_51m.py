#!/usr/bin/env python3
"""Pure metric implementation for frozen mei-51m longitudinal evaluations.

Prediction producers may be Python/MLX or Browser-WASM.  This scorer is
runtime-neutral and refuses incomplete sample coverage, duplicate IDs,
single-class confidence outcomes and drifted evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import sft_v4_contract_51m as contract


CALL_ID = re.compile(r"^call-s[0-9a-f]{8}-[1-8]-[0-9a-f]{12}$")
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def _prediction_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in result:
            raise RuntimeError(f"empty or duplicate prediction sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _join_predictions(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id = _prediction_map(predictions)
    gold_ids = [str(row.get("sample_id") or "") for row in gold]
    if len(gold_ids) != len(set(gold_ids)) or any(not value for value in gold_ids):
        raise RuntimeError("gold bank contains empty or duplicate sample IDs")
    missing = sorted(set(gold_ids) - set(by_id))
    extra = sorted(set(by_id) - set(gold_ids))
    if missing or extra:
        raise RuntimeError(
            f"prediction coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )
    return [(row, by_id[str(row["sample_id"])]) for row in gold]


def retrieval_metrics(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    joined = _join_predictions(gold, predictions)
    recalls_1: list[float] = []
    recalls_5: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_5: list[float] = []
    lexical_5: list[float] = []
    by_family: dict[str, list[float]] = defaultdict(list)
    by_tool: dict[str, list[float]] = defaultdict(list)
    for row, prediction in joined:
        target = str(row["gold_tool"])
        ranked = [str(value) for value in prediction.get("ranked_tools") or []]
        lexical = [str(value) for value in prediction.get("lexical_ranked_tools") or []]
        if len(ranked) < 5 or len(set(ranked)) != len(ranked):
            raise RuntimeError(f"invalid learned ranking: {row['sample_id']}")
        rank = ranked.index(target) + 1 if target in ranked else None
        hit1 = float(rank == 1)
        hit5 = float(rank is not None and rank <= 5)
        recalls_1.append(hit1)
        recalls_5.append(hit5)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        ndcg_5.append(1.0 / math.log2(rank + 1) if rank and rank <= 5 else 0.0)
        lexical_5.append(float(target in lexical[:5]))
        by_family[str(row.get("family") or "unknown")].append(hit5)
        by_tool[target].append(hit5)
    learned = _mean(recalls_5)
    lexical = _mean(lexical_5)
    return {
        "n": len(joined),
        "recall_at_1": _mean(recalls_1),
        "recall_at_5": learned,
        "mrr": _mean(reciprocal_ranks),
        "ndcg_at_5": _mean(ndcg_5),
        "lexical_recall_at_5": lexical,
        "learned_minus_lexical_recall_at_5": learned - lexical,
        "by_family_recall_at_5": {
            key: _mean(value) for key, value in sorted(by_family.items())
        },
        "by_tool_recall_at_5": {
            key: _mean(value) for key, value in sorted(by_tool.items())
        },
    }


def _turn_variant(prediction: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    raw = prediction.get("result", prediction)
    if not isinstance(raw, dict):
        return "error", None
    kind = str(raw.get("kind") or "")
    if kind == "call":
        call = raw.get("call")
        if isinstance(call, dict):
            return kind, call
        calls = raw.get("calls")
        if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
            return kind, calls[0]
        return "error", None
    if kind in {"refuse", "respond", "error"}:
        return kind, None
    calls = raw.get("calls")
    if isinstance(calls, list):
        if len(calls) == 1 and isinstance(calls[0], dict):
            return "call", calls[0]
        if not calls:
            return "refuse", None
    return "error", None


def fullcall_metrics(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    joined = _join_predictions(gold, predictions)
    by_name = {str(tool["name"]): tool for tool in tools}
    execute_n = 0
    refuse_n = 0
    execute_call = 0
    execute_refuse = 0
    refuse_correct = 0
    refuse_call = 0
    name_exact = 0
    args_exact = 0
    schema_valid = 0
    errors = 0
    reason_scores: dict[str, list[float]] = defaultdict(list)
    for row, prediction in joined:
        variant, call = _turn_variant(prediction)
        is_execute = row.get("kind") == "execute"
        if is_execute:
            execute_n += 1
            if variant == "call" and call is not None:
                execute_call += 1
                predicted_name = str(call.get("name") or "")
                predicted_args = call.get("arguments")
                expected_name = str(row.get("gold_name") or "")
                expected_args = row.get("gold_args") or {}
                if predicted_name == expected_name:
                    name_exact += 1
                    if predicted_args == expected_args:
                        args_exact += 1
                tool = by_name.get(predicted_name)
                if tool and contract.arguments_match_schema(
                    predicted_args, tool.get("parameters") or {}
                ):
                    schema_valid += 1
            elif variant == "refuse":
                execute_refuse += 1
            else:
                errors += 1
        else:
            refuse_n += 1
            correct = variant == "refuse"
            refuse_correct += int(correct)
            refuse_call += int(variant == "call")
            errors += int(variant not in {"refuse", "call"})
            reason_scores[str(row.get("reason_code") or "unknown")].append(float(correct))
    if not execute_n or not refuse_n:
        raise RuntimeError("fullcall bank must contain both execute and refuse rows")
    execute_detection = execute_call / execute_n
    refusal_accuracy = refuse_correct / refuse_n
    return {
        "n": len(joined),
        "execute_n": execute_n,
        "refuse_n": refuse_n,
        "execute_detection_accuracy": execute_detection,
        "execute_tool_name_exact": name_exact / execute_n,
        "execute_arguments_exact": args_exact / execute_n,
        "execute_schema_valid": schema_valid / execute_n,
        "refusal_accuracy": refusal_accuracy,
        "false_refuse_rate": execute_refuse / execute_n,
        "false_execute_rate": refuse_call / refuse_n,
        "balanced_accuracy": (execute_detection + refusal_accuracy) / 2.0,
        "error_rate": errors / len(joined),
        "refusal_accuracy_by_reason": {
            key: _mean(value) for key, value in sorted(reason_scores.items())
        },
        "baselines": {
            "all_refuse": {
                "execute_detection_accuracy": 0.0,
                "refusal_accuracy": 1.0,
                "balanced_accuracy": 0.5,
            },
            "always_first_tool": {
                "execute_tool_name_exact": _mean(
                    float((row.get("retrieved_tools") or [None])[0] == row.get("gold_name"))
                    for row in gold
                    if row.get("kind") == "execute"
                ),
                "refusal_accuracy": 0.0,
            },
        },
    }


def classification_metrics(
    labels: list[int], predictions: list[int], *, n_classes: int
) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise RuntimeError("classification labels/predictions length mismatch")
    confusion = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for gold, predicted in zip(labels, predictions):
        if not 0 <= gold < n_classes or not 0 <= predicted < n_classes:
            raise RuntimeError("classification label outside class range")
        confusion[gold][predicted] += 1
    per_class: list[dict[str, Any]] = []
    for class_id in range(n_classes):
        tp = confusion[class_id][class_id]
        fp = sum(confusion[row][class_id] for row in range(n_classes)) - tp
        fn = sum(confusion[class_id]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "support": sum(confusion[class_id]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    negatives = sum(1 for value in labels if value != 0)
    false_continue = sum(
        1 for gold, predicted in zip(labels, predictions) if gold != 0 and predicted == 0
    )
    return {
        "n": len(labels),
        "accuracy": _mean(float(a == b) for a, b in zip(labels, predictions)),
        "macro_f1": _mean(item["f1"] for item in per_class),
        "per_class": per_class,
        "confusion": confusion,
        "class_0_false_continue_rate": false_continue / negatives if negatives else 0.0,
    }


def mw_metrics(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    joined = _join_predictions(gold, predictions)
    labels = [int(row[0]["reason_class_id"]) for row in joined]
    predicted = [int(row[1]["predicted_class_id"]) for row in joined]
    return classification_metrics(labels, predicted, n_classes=20)


def _auroc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        raise RuntimeError("confidence evaluation requires positive and negative outcomes")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def _auprc(labels: list[int], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    positives = sum(labels)
    if not positives:
        raise RuntimeError("confidence evaluation has no positive outcomes")
    seen_positive = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index] == 1:
            seen_positive += 1
            precision_sum += seen_positive / rank
    return precision_sum / positives


def confidence_metrics(
    rows: list[dict[str, Any]], *, minimum_class_rows: int = 1
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    if set(labels) != {0, 1}:
        raise RuntimeError("confidence evaluation is single-class")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives < minimum_class_rows or negatives < minimum_class_rows:
        raise RuntimeError(
            "confidence evaluation below minimum class coverage: "
            f"positive={positives} negative={negatives} floor={minimum_class_rows}"
        )
    if any(not 0.0 <= value <= 1.0 for value in scores):
        raise RuntimeError("confidence score outside [0,1]")
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        selected = [
            offset
            for offset, score in enumerate(scores)
            if low <= score <= high and (index == 9 or score < high)
        ]
        if not selected:
            bins.append({"low": low, "high": high, "n": 0})
            continue
        confidence = _mean(scores[offset] for offset in selected)
        accuracy = _mean(labels[offset] for offset in selected)
        ece += len(selected) / len(rows) * abs(confidence - accuracy)
        bins.append(
            {
                "low": low,
                "high": high,
                "n": len(selected),
                "mean_score": confidence,
                "accuracy": accuracy,
            }
        )
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    risk_coverage = {}
    for coverage in (0.25, 0.5, 0.75, 1.0):
        count = max(1, math.ceil(len(rows) * coverage))
        risk_coverage[f"{coverage:.2f}"] = 1.0 - _mean(labels[index] for index in order[:count])
    return {
        "n": len(rows),
        "positive_n": positives,
        "negative_n": negatives,
        "auroc": _auroc(labels, scores),
        "auprc": _auprc(labels, scores),
        "ece_10": ece,
        "brier": _mean((score - label) ** 2 for score, label in zip(scores, labels)),
        "calibration_bins": bins,
        "risk_coverage": risk_coverage,
    }


def confidence_outcome_metrics(
    candidates: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    minimum_class_rows: int = 1,
) -> dict[str, Any]:
    """Score confidence only after exact frozen-candidate identity matching."""

    joined = _join_predictions(candidates, predictions)
    scored: list[dict[str, Any]] = []
    for candidate, prediction in joined:
        for field in ("source_sample_id", "expected_kind", "candidate_tool"):
            if prediction.get(field) != candidate.get(field):
                raise RuntimeError(
                    "confidence candidate metadata mismatch: "
                    f"sample_id={candidate['sample_id']} field={field}"
                )
        label = prediction.get("label")
        if isinstance(label, bool) or label not in (0, 1):
            raise RuntimeError(
                f"invalid confidence outcome label: {candidate['sample_id']}"
            )
        score = prediction.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RuntimeError(
                f"invalid confidence outcome score: {candidate['sample_id']}"
            )
        scored.append({"label": int(label), "score": float(score)})
    report = confidence_metrics(scored, minimum_class_rows=minimum_class_rows)
    report["candidate_identity_verified"] = True
    return report


def _polarity_match(text: str, polarity: str) -> bool:
    if polarity.startswith("failure"):
        return "失败" in text or "未成功" in text
    if polarity == "cancelled":
        return "取消" in text
    if polarity == "no_change":
        return "无需调整" in text or "没有变化" in text
    if polarity == "partial":
        return "部分" in text and ("失败" in text or "未成功" in text)
    if polarity == "success_off":
        return "关闭" in text
    if polarity == "success_on":
        return "启动" in text or "开启" in text
    if polarity == "success_lock":
        return "锁定" in text
    return bool(text.strip())


def narration_metrics(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    joined = _join_predictions(gold, predictions)
    fact_scores: list[float] = []
    number_scores: list[float] = []
    polarity_scores: list[float] = []
    fallback_scores: list[float] = []
    delivered: list[float] = []
    for row, prediction in joined:
        text = str(prediction.get("text") or "")
        facts = [str(value) for value in row.get("required_facts") or []]
        fact = _mean(float(value in text) for value in facts) if facts else 1.0
        expected_numbers = NUMBER.findall(str(row.get("target") or ""))
        actual_numbers = NUMBER.findall(text)
        number = float(all(value in actual_numbers for value in expected_numbers))
        polarity = float(_polarity_match(text, str(row.get("polarity") or "")))
        fallback_text = str(prediction.get("fallback_text") or "")
        fallback = float(fallback_text == str(row.get("target") or ""))
        fact_scores.append(fact)
        number_scores.append(number)
        polarity_scores.append(polarity)
        fallback_scores.append(fallback)
        delivered.append(float(fact == 1.0 and number == 1.0 and polarity == 1.0))
    return {
        "n": len(joined),
        "required_fact_recall": _mean(fact_scores),
        "number_preservation": _mean(number_scores),
        "polarity_accuracy": _mean(polarity_scores),
        "fallback_exact": _mean(fallback_scores),
        "delivered_answer_correctness": _mean(delivered),
    }


def multistep_metrics(
    gold: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    joined = _join_predictions(gold, predictions)
    trajectory_steps: dict[str, list[bool]] = defaultdict(list)
    trajectory_lengths: dict[str, int] = {}
    step_scores: list[float] = []
    call_ids: list[str] = []
    terminal_scores: list[float] = []
    for row, prediction in joined:
        variant, call = _turn_variant(prediction)
        if row.get("kind") == "execute":
            answers = row.get("answers") or []
            expected = answers[0] if answers else {}
            correct = bool(
                variant == "call"
                and call
                and call.get("name") == expected.get("name")
                and call.get("arguments") == expected.get("arguments")
            )
            if call:
                call_ids.append(str(call.get("call_id") or ""))
        else:
            correct = variant == "respond"
            terminal_scores.append(float(correct))
        step_scores.append(float(correct))
        trajectory = str(row.get("trajectory_id") or row.get("cf_group") or "")
        trajectory_steps[trajectory].append(correct)
        trajectory_lengths[trajectory] = int(row.get("trajectory_length") or 0)
    id_integrity = bool(
        len(call_ids) == len(set(call_ids)) and all(CALL_ID.fullmatch(value) for value in call_ids)
    )
    by_length: dict[str, list[float]] = defaultdict(list)
    successes = []
    for trajectory, steps in trajectory_steps.items():
        success = float(all(steps))
        successes.append(success)
        by_length[str(trajectory_lengths[trajectory])].append(success)
    return {
        "n_steps": len(joined),
        "n_trajectories": len(trajectory_steps),
        "trajectory_success": _mean(successes),
        "step_accuracy": _mean(step_scores),
        "call_id_integrity": float(id_integrity),
        "terminal_response_accuracy": _mean(terminal_scores),
        "by_trajectory_length": {
            key: _mean(value) for key, value in sorted(by_length.items())
        },
    }


def verify_lock(lock_dir: Path) -> dict[str, Any]:
    lock = contract.load_json(lock_dir / "lock.json")
    if (
        lock.get("schema")
        not in {
            "mei-51m-longitudinal-eval-lock-v3",
            "mei-51m-longitudinal-eval-lock-v4",
        }
        or not str(lock.get("id") or "").startswith("mei-51m-longitudinal-eval-v")
        or lock.get("status") != "frozen"
    ):
        raise RuntimeError("not a frozen mei-51m longitudinal lock")
    for name, spec in (lock.get("artifacts") or {}).items():
        path = lock_dir / name
        if not path.is_file() or contract.sha_file(path) != spec.get("sha256"):
            raise RuntimeError(f"longitudinal artifact hash drift: {path}")
    isolation = contract.load_json(lock_dir / "isolation-receipt.json")
    if isolation.get("status") != "passed":
        raise RuntimeError("longitudinal isolation receipt did not pass")
    return lock


def _load_prediction(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing prediction file: {path}")
    return contract.load_jsonl(path)


def score_split(lock_dir: Path, prediction_dir: Path, split: str) -> dict[str, Any]:
    lock = verify_lock(lock_dir)
    tools = contract.universe_tools(lock_dir / "tool-universe.json")
    result: dict[str, Any] = {
        "retrieval": retrieval_metrics(
            contract.load_jsonl(lock_dir / f"retrieval.{split}.jsonl"),
            _load_prediction(prediction_dir / f"retrieval.{split}.jsonl"),
        ),
        "fullcall": fullcall_metrics(
            contract.load_jsonl(lock_dir / f"fullcall.{split}.jsonl"),
            _load_prediction(prediction_dir / f"fullcall.{split}.jsonl"),
            tools,
        ),
        "mw_disposition": mw_metrics(
            contract.load_jsonl(lock_dir / f"mw.{split}.jsonl"),
            _load_prediction(prediction_dir / f"mw.{split}.jsonl"),
        ),
        "multi_step": multistep_metrics(
            contract.load_jsonl(lock_dir / f"multistep.{split}.jsonl"),
            _load_prediction(prediction_dir / f"multistep.{split}.jsonl"),
        ),
        "narration": narration_metrics(
            contract.load_jsonl(lock_dir / f"narration.{split}.jsonl"),
            _load_prediction(prediction_dir / f"narration.{split}.jsonl"),
        ),
        "confidence": confidence_outcome_metrics(
            contract.load_jsonl(lock_dir / f"confidence.{split}.jsonl"),
            _load_prediction(prediction_dir / f"confidence.{split}.jsonl"),
            minimum_class_rows=100,
        ),
    }
    if lock.get("schema") == "mei-51m-longitudinal-eval-lock-v4":
        holdout_document = contract.load_json(
            lock_dir / "schema-holdout-tool-universe.json"
        )
        schema_tools = [
            *tools,
            *[
                contract.compact_tool(tool)
                for tool in holdout_document.get("tools") or []
            ],
        ]
        if len({str(tool["name"]) for tool in schema_tools}) != 179:
            raise RuntimeError("eval-v7 schema holdout catalog drifted")
        result["natural_cross_generator"] = {
            "retrieval": retrieval_metrics(
                contract.load_jsonl(
                    lock_dir / f"natural-retrieval.{split}.jsonl"
                ),
                _load_prediction(
                    prediction_dir / f"natural-retrieval.{split}.jsonl"
                ),
            ),
            "fullcall": fullcall_metrics(
                contract.load_jsonl(lock_dir / f"natural-fullcall.{split}.jsonl"),
                _load_prediction(
                    prediction_dir / f"natural-fullcall.{split}.jsonl"
                ),
                tools,
            ),
        }
        result["whole_schema_holdout"] = {
            "retrieval": retrieval_metrics(
                contract.load_jsonl(lock_dir / f"schema-retrieval.{split}.jsonl"),
                _load_prediction(
                    prediction_dir / f"schema-retrieval.{split}.jsonl"
                ),
            ),
            "fullcall": fullcall_metrics(
                contract.load_jsonl(lock_dir / f"schema-fullcall.{split}.jsonl"),
                _load_prediction(
                    prediction_dir / f"schema-fullcall.{split}.jsonl"
                ),
                schema_tools,
            ),
            "confidence": confidence_outcome_metrics(
                contract.load_jsonl(
                    lock_dir / f"schema-confidence.{split}.jsonl"
                ),
                _load_prediction(
                    prediction_dir / f"schema-confidence.{split}.jsonl"
                ),
                minimum_class_rows=1,
            ),
        }
    return {
        "schema": "mei-51m-longitudinal-scorecard-v3",
        "split": split,
        "evaluation_fingerprint": lock["evaluation_fingerprint"],
        "metrics": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=contract.DEFAULT_EVAL_ROOT / contract.EVAL_ID,
    )
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scorecard = score_split(args.lock_dir, args.predictions_dir, args.split)
    contract.write_json_once(args.out, scorecard)
    print(json.dumps(scorecard, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
