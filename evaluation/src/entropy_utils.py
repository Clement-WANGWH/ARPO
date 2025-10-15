import json
import math
import os
from typing import Any, Dict, List, Optional

MODULE_TAGS = ["think", "python", "result", "answer"]


def _as_dict(obj: Any) -> Dict[str, float]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {str(k): float(v) for k, v in obj.items() if v is not None}
    if hasattr(obj, "items"):
        return {str(k): float(v) for k, v in obj.items() if v is not None}
    return {}


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def compute_entropy_from_top_logprobs(
    top_logprob_entry: Dict[str, float]
) -> Optional[float]:
    if not top_logprob_entry:
        return None
    values = list(top_logprob_entry.values())
    if not values:
        return None
    max_logprob = max(values)
    exp_values = [math.exp(v - max_logprob) for v in values]
    total = sum(exp_values)
    if total <= 0:
        return None
    probs = [v / total for v in exp_values]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return float(entropy)


def extract_token_stats_from_choice(choice: Any) -> Dict[str, Any]:
    logprobs = getattr(choice, "logprobs", None)
    tokens = _get_attr(logprobs, "tokens", []) or []
    token_logprobs = _get_attr(logprobs, "token_logprobs", []) or []
    top_logprobs_raw = _get_attr(logprobs, "top_logprobs", []) or []

    top_logprobs: List[Dict[str, float]] = []
    entropies: List[Optional[float]] = []
    for entry in top_logprobs_raw:
        top_dict = _as_dict(entry)
        top_logprobs.append(top_dict)
        entropies.append(compute_entropy_from_top_logprobs(top_dict))

    length = min(len(tokens), len(token_logprobs), len(top_logprobs), len(entropies))
    tokens = [str(tok) for tok in tokens[:length]]
    token_logprobs = [
        (float(lp) if lp is not None else None) for lp in token_logprobs[:length]
    ]
    top_logprobs = top_logprobs[:length]
    entropies = entropies[:length]

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "entropies": entropies,
        "text": getattr(choice, "text", ""),
    }


def trim_token_stats_to_match_text(
    record: Dict[str, Any], target_text: str
) -> Dict[str, Any]:
    if not record:
        return record
    if not target_text:
        return {
            **record,
            "tokens": [],
            "entropies": [],
            "token_logprobs": [],
            "top_logprobs": [],
            "text": target_text,
        }
    tokens: List[str] = record.get("tokens", [])
    entropies: List[Optional[float]] = record.get("entropies", [])
    token_logprobs: List[Optional[float]] = record.get("token_logprobs", [])
    top_logprobs: List[Dict[str, float]] = record.get("top_logprobs", [])

    if not tokens:
        return {**record, "text": target_text}

    trimmed_tokens: List[str] = []
    trimmed_entropies: List[Optional[float]] = []
    trimmed_logprobs: List[Optional[float]] = []
    trimmed_top_logprobs: List[Dict[str, float]] = []

    current_text = ""
    for token, entropy, logprob, top in zip(
        tokens, entropies, token_logprobs, top_logprobs
    ):
        candidate = current_text + token
        if target_text and not target_text.startswith(candidate):
            break
        trimmed_tokens.append(token)
        trimmed_entropies.append(entropy)
        trimmed_logprobs.append(logprob)
        trimmed_top_logprobs.append(top)
        current_text = candidate
        if current_text == target_text:
            break

    if target_text and "".join(trimmed_tokens) != target_text:
        return {
            **record,
            "tokens": [],
            "entropies": [],
            "token_logprobs": [],
            "top_logprobs": [],
            "text": target_text,
        }

    return {
        "tokens": trimmed_tokens,
        "entropies": trimmed_entropies,
        "token_logprobs": trimmed_logprobs,
        "top_logprobs": trimmed_top_logprobs,
        "text": target_text,
    }


def _token_char_bounds(tokens: List[str]) -> List[Dict[str, int]]:
    bounds: List[Dict[str, int]] = []
    current = 0
    for token in tokens:
        start = current
        current += len(token)
        bounds.append({"start": start, "end": current})
    return bounds


def _char_to_token_start(bounds: List[Dict[str, int]], char_pos: int) -> int:
    for idx, bound in enumerate(bounds):
        if char_pos < bound["end"]:
            return idx
    return max(len(bounds) - 1, 0) if bounds else 0


def _char_to_token_end(bounds: List[Dict[str, int]], char_pos: int) -> int:
    for idx, bound in enumerate(bounds):
        if char_pos <= bound["start"]:
            return idx
    return len(bounds)


def compute_module_segments(text: str, tokens: List[str]) -> List[Dict[str, Any]]:
    if not text:
        return []
    bounds = _token_char_bounds(tokens)
    segments: List[Dict[str, Any]] = []
    for tag in MODULE_TAGS:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        search_start = 0
        while True:
            start_idx = text.find(open_tag, search_start)
            if start_idx == -1:
                break
            end_idx = text.find(close_tag, start_idx + len(open_tag))
            if end_idx == -1:
                break
            end_idx += len(close_tag)
            start_token = _char_to_token_start(bounds, start_idx)
            end_token = _char_to_token_end(bounds, end_idx)
            segments.append(
                {
                    "tag": tag,
                    "start_char": start_idx,
                    "end_char": end_idx,
                    "start_token": start_token,
                    "end_token": end_token,
                }
            )
            search_start = end_idx
    segments.sort(key=lambda item: item["start_char"])
    return segments


def aggregate_token_records(
    records: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    tokens: List[str] = []
    entropies: List[Optional[float]] = []
    token_logprobs: List[Optional[float]] = []
    top_logprobs: List[Dict[str, float]] = []
    texts: List[str] = []
    for record in records:
        if not record:
            continue
        record_tokens = record.get("tokens", []) or []
        record_entropies = record.get("entropies", []) or []
        record_logprobs = record.get("token_logprobs", []) or []
        record_top = record.get("top_logprobs", []) or []
        length = min(
            len(record_tokens), len(record_entropies), len(record_logprobs), len(record_top)
        )
        tokens.extend(record_tokens[:length])
        entropies.extend(record_entropies[:length])
        token_logprobs.extend(record_logprobs[:length])
        top_logprobs.extend(record_top[:length])
        texts.append(record.get("text", ""))

    full_text = "".join(texts)
    token_text = "".join(tokens)
    modules = (
        compute_module_segments(full_text, tokens)
        if full_text and token_text and len(token_text) == len(full_text)
        else []
    )
    return {
        "tokens": tokens,
        "entropies": entropies,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text": full_text,
        "modules": modules,
    }


def save_entropy_plot(
    tokens: List[str],
    entropies: List[Optional[float]],
    modules: List[Dict[str, Any]],
    output_file: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not entropies:
        return

    xs = list(range(len(entropies)))
    y_values = [float("nan") if e is None else float(e) for e in entropies]
    if all(math.isnan(v) for v in y_values):
        return

    fig_width = max(8, len(entropies) * 0.2)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.plot(xs, y_values, color="#1f77b4", linewidth=1.5, marker="o", markersize=2)

    colors = {
        "think": "#6baed6",
        "python": "#fd8d3c",
        "result": "#74c476",
        "answer": "#ef3b2c",
    }

    finite_values = [v for v in y_values if not math.isnan(v)]
    if finite_values:
        min_val = min(finite_values)
        max_val = max(finite_values)
    else:
        min_val = 0.0
        max_val = 1.0
    value_span = max(max_val - min_val, 1e-6)
    padding = 0.15 * value_span

    legend_used = set()
    for segment in modules:
        tag = segment.get("tag", "")
        start_token = segment.get("start_token", 0)
        end_token = segment.get("end_token", start_token)
        color = colors.get(tag, "#9e9ac8")
        label = f"<{tag}>" if tag and tag not in legend_used else None
        ax.axvspan(start_token, end_token, color=color, alpha=0.18, label=label)
        if label:
            legend_used.add(tag)
    if legend_used:
        ax.legend(loc="upper right")

    text_y = max_val + padding
    for segment in modules:
        tag = segment.get("tag", "")
        if not tag:
            continue
        start_token = segment.get("start_token", 0)
        end_token = segment.get("end_token", start_token)
        center = (start_token + end_token) / 2 if end_token != start_token else start_token
        color = colors.get(tag, "#555555")
        ax.text(
            center,
            text_y,
            f"<{tag}>",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=color,
        )

    ax.set_xlabel("Token Index")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Token-level Entropy Trajectory")
    ax.set_xlim(0, max(xs) if xs else 1)
    ax.set_ylim(min_val - padding, text_y + padding)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def save_sample_artifacts(
    sample_stat: Dict[str, Any], sample_dir: str, sample_index: int
) -> None:
    os.makedirs(sample_dir, exist_ok=True)
    trajectory_path = os.path.join(sample_dir, "trajectory.txt")
    metadata_path = os.path.join(sample_dir, "metadata.json")
    token_trace_path = os.path.join(sample_dir, "token_trace.json")
    plot_path = os.path.join(sample_dir, "token_entropy.png")

    with open(trajectory_path, "w", encoding="utf-8") as f:
        f.write(f"Sample Index: {sample_index}\n")
        f.write(f"Instruction: {sample_stat.get('instruction', '')}\n")
        f.write(f"Question: {sample_stat.get('input', '')}\n")
        f.write(f"Golden Answer: {sample_stat.get('answer', '')}\n")
        f.write(f"Prediction: {sample_stat.get('prediction', '')}\n\n")
        f.write("Full Output:\n")
        f.write(sample_stat.get("output", ""))
        f.write("\n\n-- Logs --\n")
        for log in sample_stat.get("logs", []):
            f.write(f"{log}\n")

    metadata = {
        "sample_index": sample_index,
        "question": sample_stat.get("input"),
        "golden_answer": sample_stat.get("answer"),
        "prediction": sample_stat.get("prediction"),
        "timing": sample_stat.get("timing"),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    token_trace = sample_stat.get("token_trace")
    if token_trace:
        with open(token_trace_path, "w", encoding="utf-8") as f:
            json.dump(token_trace, f, indent=2, ensure_ascii=False)
        save_entropy_plot(
            token_trace.get("tokens", []),
            token_trace.get("entropies", []),
            token_trace.get("modules", []),
            plot_path,
        )
