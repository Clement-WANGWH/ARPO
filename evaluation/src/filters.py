import os
import json
from typing import Iterable, Dict, Any, Tuple


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip malformed lines
                continue


def _is_level_45(entry: Dict[str, Any]) -> bool:
    level = entry.get("level") or entry.get("Level") or entry.get("difficulty")
    if not level:
        return False
    level_str = str(level)
    return ("Level 4" in level_str) or ("Level 5" in level_str)


def _is_standard_math_answer(entry: Dict[str, Any]) -> bool:
    # Filter out obviously invalid or non-math entries
    ans = entry.get("answer")
    if ans is None:
        return False
    ans_s = str(ans).strip()
    if ans_s == "" or ans_s.lower() in {"none", "null"}:
        return False
    # Accept LaTeX, numeric, short algebraic strings; reject long sentences or JSON/arrays
    if isinstance(ans, (list, dict)):
        return False
    # Heuristic: too many spaces indicates full-sentence text rather than a compact math answer
    if ans_s.count(" ") > 6:
        return False
    return True


def build_math500_level45(
    math_jsonl_path: str,
    out_jsonl_path: str,
) -> Tuple[int, int]:
    """
    Build a filtered subset (Level 4 & 5) from MATH `test.jsonl` into a math500-style file.

    Returns (kept, total_scanned).
    """
    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    kept = 0
    total = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as out_f:
        for item in _iter_jsonl(math_jsonl_path):
            total += 1
            if not _is_level_45(item):
                continue
            if not _is_standard_math_answer(item):
                continue
            # Normalize to a consistent schema: question/answer
            q = item.get("problem") or item.get("question") or ""
            a = item.get("answer") or ""
            if not q or not a:
                continue
            out_f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
            kept += 1
    return kept, total

