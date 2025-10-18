import json
import math
import os
from typing import Any, Dict, List, Optional

# Only mark the requested stages on the plot
MODULE_TAGS = ["think", "python", "search", "result", "answer"]


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


def _normalize_for_match(s: str) -> str:
    if not s:
        return s
    # Unify newlines and strip zero-width/BOM characters that often appear in decoding
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\uFEFF", "").replace("\ufeff", "")  # BOM
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")  # ZW* chars
    return s


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

    # First try: strict prefix alignment on normalized text
    norm_target = _normalize_for_match(target_text)
    current_text = ""
    success = False
    for token, entropy, logprob, top in zip(tokens, entropies, token_logprobs, top_logprobs):
        candidate = current_text + token
        if norm_target and not _normalize_for_match(norm_target).startswith(_normalize_for_match(candidate)):
            # stop at the first mismatch
            break
        trimmed_tokens.append(token)
        trimmed_entropies.append(entropy)
        trimmed_logprobs.append(logprob)
        trimmed_top_logprobs.append(top)
        current_text = candidate
        if _normalize_for_match(current_text) == norm_target:
            success = True
            break

    # Fallback 1: if nothing matched but we do have tokens, try length-based cutoff
    if not success and not trimmed_tokens and tokens:
        current_text = ""
        for token, entropy, logprob, top in zip(tokens, entropies, token_logprobs, top_logprobs):
            trimmed_tokens.append(token)
            trimmed_entropies.append(entropy)
            trimmed_logprobs.append(logprob)
            trimmed_top_logprobs.append(top)
            current_text += token
            if len(_normalize_for_match(current_text)) >= len(norm_target):
                success = True
                break

    # Final verification; if still not aligned, keep the original target_text but avoid dropping everything
    if target_text and _normalize_for_match("".join(trimmed_tokens))[: len(norm_target)] != norm_target:
        # Keep what we have if we matched at least one token, else fall back to empty to avoid misleading alignment
        if not trimmed_tokens:
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
    # Compute module segments against token_text to ensure alignment
    modules = compute_module_segments(token_text, tokens) if tokens else []
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

    # Make x-axis more compact and cap overall width for readability
    fig_width = min(16, max(6, len(entropies) * 0.06))
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    ax.plot(xs, y_values, color="#1f77b4", linewidth=1.0)

    colors = {
        "think": "#6baed6",
        "python": "#fd8d3c",
        "search": "#9467bd",
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
    # Thin out ticks for dense sequences
    if len(xs) > 100:
        step = max(1, len(xs) // 10)
        ax.set_xticks(list(range(0, len(xs), step)))
    ax.set_ylim(min_val - padding, text_y + padding)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def save_token_entropy_text_image(
    tokens: List[str],
    entropies: List[Optional[float]],
    output_file: str,
    max_width_in: float = 12.0,
    line_height: float = 0.16,
    tokens_per_row: int = 15,
    rows_per_page: int = 20,
) -> List[str]:
    """Render token entropy heatmap text as one or more pages.

    Changes from previous behavior:
    - Uses variable-width highlights sized by the token's text length.
    - Paginates to cover the full generation; each page has a fixed size.
    - Wraps tokens per row by available width instead of a fixed count per row.

    Notes:
    - `tokens_per_row` is kept for backward compatibility but is ignored for layout.
    - Returns a list of generated file paths. When multiple pages are created, files
      are suffixed with _0001, _0002, ...; otherwise a single .png is produced.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.patches import Rectangle
    except ImportError:
        return

    if not tokens or not entropies:
        return []

    # Normalize entropy values to [0, 1]
    vals = [e for e in entropies if e is not None and not math.isnan(e)]
    if not vals:
        return []
    vmin, vmax = min(vals), max(vals)
    span = max(vmax - vmin, 1e-6)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.get_cmap("coolwarm")  # blue -> red

    # Build rows with fixed tokens per row to avoid crowding
    n = len(tokens)
    tokens_per_row = max(5, int(tokens_per_row))
    rows_per_page = max(5, int(rows_per_page))
    rows = [
        (tokens[i : i + tokens_per_row], entropies[i : i + tokens_per_row])
        for i in range(0, n, tokens_per_row)
    ]
    n_rows_total = len(rows)

    # Figure sizing: fixed size for every page
    dpi_local = 200
    width_in = float(max_width_in)
    height_in = 8.0
    width_px = int(width_in * dpi_local)
    height_px = int(height_in * dpi_local)

    # Layout parameters (in pixels)
    left_pad_px = int(0.02 * width_px)
    right_pad_px = int(0.02 * width_px)
    top_pad_px = int(0.06 * height_px)
    bottom_pad_px = int(0.18 * height_px)
    usable_w_px = max(1, width_px - left_pad_px - right_pad_px)
    usable_h_px = max(1, height_px - top_pad_px - bottom_pad_px)

    # Row height and spacing (consistent per page)
    row_gap_px = max(4, int(0.012 * height_px))
    row_height_px = max(14, int((usable_h_px - (rows_per_page - 1) * row_gap_px) / rows_per_page))
    baseline_offset_px = int(0.56 * row_height_px)

    # Cell width
    cell_w_px = usable_w_px / tokens_per_row

    # Token width estimate parameters
    fontsize = 9
    approx_char_px = 9  # monospace estimate at 200 DPI
    token_hpad_px = max(8, int(0.12 * cell_w_px))
    token_vpad_px = max(2, int(0.10 * row_height_px))

    # Pagination
    pages = max(1, (n_rows_total + rows_per_page - 1) // rows_per_page)
    page_files: List[str] = []
    prefix = output_file[:-4] if output_file.lower().endswith(".png") else output_file

    for p in range(pages):
        start_row = p * rows_per_page
        end_row = min(n_rows_total, (p + 1) * rows_per_page)
        page_rows = rows[start_row:end_row]

        fig, ax = plt.subplots(figsize=(width_in, height_in), constrained_layout=False)
        ax.set_axis_off()

        for r, (row_tokens, row_ents) in enumerate(page_rows):
            y_center_px = height_px - top_pad_px - baseline_offset_px - r * (row_height_px + row_gap_px)
            for c, (t, e) in enumerate(zip(row_tokens, row_ents)):
                if e is None or math.isnan(e):
                    e = vmin
                color = cmap(norm(e))

                # Prepare text and capacity per cell
                display_text = t.replace("$", r"\$").replace("\n", "↵")
                max_text_px = cell_w_px - 2 * token_hpad_px
                max_chars_fit = max(1, int(max_text_px / max(1, approx_char_px)))
                if len(display_text) > max_chars_fit:
                    display_text = display_text[: max(1, max_chars_fit - 1)] + "…"

                token_text_px = len(display_text) * approx_char_px
                rect_w_px = min(cell_w_px * 0.98, token_text_px + 2 * token_hpad_px)

                # Positioning of the cell
                x0_px = left_pad_px + c * cell_w_px + (cell_w_px - rect_w_px) / 2
                rect_h_px = max(6, int(row_height_px * 0.82))
                rect_y0_px = int(y_center_px - rect_h_px / 2)

                # Draw rectangle
                rect = Rectangle(
                    (x0_px / width_px, rect_y0_px / height_px),
                    rect_w_px / width_px,
                    rect_h_px / height_px,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor='none',
                    alpha=0.95,
                )
                ax.add_patch(rect)

                # Draw text centered
                # Choose text color by background luminance: deep blue/red => white; light => black
                r_, g_, b_, _ = color
                luminance = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_
                txt_color = 'white' if luminance < 0.5 else 'black'

                ax.text(
                    (x0_px + rect_w_px / 2) / width_px,
                    (rect_y0_px + rect_h_px / 2) / height_px,
                    display_text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontfamily="monospace",
                    fontweight="bold",
                    color=txt_color,
                    transform=ax.transAxes,
                )

        # Colorbar
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02, fraction=0.06)
        cb.set_label('Token Entropy (nats) — low→blue, high→red')
        cb.ax.tick_params(labelsize=9)

        # Save page
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        if pages > 1:
            page_file = f"{prefix}_{p+1:04d}.png"
        else:
            page_file = f"{prefix}.png"
        fig.savefig(page_file, dpi=dpi_local, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        page_files.append(page_file)

    return page_files


def save_sample_artifacts(
    sample_stat: Dict[str, Any], sample_dir: str, sample_index: int
) -> None:
    os.makedirs(sample_dir, exist_ok=True)

    # Artifacts to keep per sample
    token_trace_path = os.path.join(sample_dir, "token_trace.json")
    plot_path = os.path.join(sample_dir, "trajectory.png")
    token_text_plot_prefix = os.path.join(sample_dir, "trajectory_tokens")

    # Write token trace and plot entropy trajectory
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
        # Token-wise background color images with pagination for readability
        page_files = save_token_entropy_text_image(
            token_trace.get("tokens", []),
            token_trace.get("entropies", []),
            token_text_plot_prefix,
        )

    # Clean up any files other than the required two artifacts
    try:
        allowed_pages = set(os.path.basename(p) for p in (page_files or []))
        allowed = {
            os.path.basename(token_trace_path),
            os.path.basename(plot_path),
            *allowed_pages,
        }
        for name in os.listdir(sample_dir):
            if name not in allowed:
                path = os.path.join(sample_dir, name)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        # Remove old paginated files if they don't belong to this run
                        if name.startswith("trajectory_tokens") and name not in allowed:
                            os.remove(path)
                        elif name in allowed:
                            pass
                        else:
                            os.remove(path)
                    elif os.path.isdir(path):
                        # Remove empty directories if created previously
                        try:
                            os.rmdir(path)
                        except OSError:
                            # Non-empty dir: skip to avoid destructive deletion
                            pass
                except Exception:
                    # Best-effort cleanup; ignore errors
                    pass
    except Exception:
        pass
