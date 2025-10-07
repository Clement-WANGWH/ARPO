#!/usr/bin/env python
"""Compare entropy trajectories between base and ARPO-tuned Qwen2.5 models.

This script follows the experimental procedure requested by the user:

* Load 20 gsm8k questions.
* Run a ReAct-style agent with a Python calculator tool using both
  Qwen2.5-7B and Qwen2.5-7B-ARPO checkpoints.
* Capture token-level logits (and thus entropies) during generation.
* Align entropy trajectories around tool feedback events and aggregate them
  into four groups (correct / incorrect for each model).
* Plot the averaged curves for comparison and print a short textual summary.

The script is intentionally self-contained and configurable via CLI flags.
It also exposes a "--dry-run" option that fabricates synthetic results in
order to validate the downstream aggregation & plotting logic without
needing GPUs or checkpoints (useful for CI sanity checks).
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import random
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


SYSTEM_PROMPT = (
    "You are an expert ReAct agent. Follow the tool-use protocol strictly.\n"
    "When solving the user's math question:\n"
    "1. Think step by step inside <think></think> tags.\n"
    "2. When calculation is required, issue <tool_call>{\"code\": "
    "<python expression>}\n</tool_call>. Only one expression per call.\n"
    "3. Wait for the tool feedback that arrives as <tool_output>{result}</tool_output>.\n"
    "4. After using tools, continue your reasoning.\n"
    "5. Once confident, reply with <final_answer>ANSWER</final_answer> and stop.\n"
    "Do not fabricate tool results. Always request computations via tool calls."
)

USER_TEMPLATE = (
    "Solve the following GSM8K math word problem.\n"
    "Return only a single final answer inside <final_answer> tags.\n"
    "Question: {question}"
)


@dataclass
class GenerationRecord:
    """Stores detailed information about a single model trajectory."""

    question: str
    target_answer: str
    model_answer: str
    full_output: str
    entropies: List[float]
    tokens: List[int]
    event_windows: List[List[float]] = field(default_factory=list)
    is_correct: bool = False


class SafeEvaluator(ast.NodeVisitor):
    """Safely evaluate arithmetic expressions for the calculator tool."""

    ALLOWED_NODES = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.Call,
        ast.Name,
    }

    SAFE_FUNCS = {
        "sqrt": math.sqrt,
        "log": math.log,
        "ln": math.log,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
    }

    SAFE_NAMES = {
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self) -> None:
        self._stack: List[ast.AST] = []

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in self.ALLOWED_NODES:
            raise ValueError(f"Disallowed expression: {type(node).__name__}")
        self._stack.append(node)
        super().generic_visit(node)
        self._stack.pop()

    def eval(self, expr: str) -> float:
        tree = ast.parse(expr, mode="eval")
        self.visit(tree)
        compiled = compile(tree, filename="<tool>", mode="eval")
        return eval(compiled, {**self.SAFE_FUNCS, **self.SAFE_NAMES}, {})


class KeywordStoppingCriteria(StoppingCriteria):
    """Stop generation once any keyword appears in the decoded text."""

    def __init__(self, tokenizer: AutoTokenizer, keywords: Sequence[str]) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.keywords = list(keywords)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        return any(keyword in text for keyword in self.keywords)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy from logits."""

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def format_dataset_answer(answer: str) -> str:
    """Normalize GSM8K answers (take the final line after '####')."""

    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def extract_final_answer(text: str) -> Optional[str]:
    match = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", text, re.S)
    if match:
        return match.group(1).strip()
    return None


def run_python_tool(code: str, evaluator: SafeEvaluator) -> str:
    try:
        value = evaluator.eval(code)
    except Exception as exc:  # noqa: BLE001 - intentionally broad for tool safety
        return json.dumps({"error": str(exc)})
    return json.dumps({"result": value})


def prepare_prompt(messages: List[str], tokenizer: AutoTokenizer) -> Tuple[torch.Tensor, int]:
    prompt = "".join(messages)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return input_ids, input_ids.shape[1]


def decode_tokens(tokenizer: AutoTokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def run_agent_on_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    target_answer: str,
    device: torch.device,
    args: argparse.Namespace,
) -> GenerationRecord:
    evaluator = SafeEvaluator()
    system_segment = f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
    user_segment = f"<|im_start|>user\n{USER_TEMPLATE.format(question=question)}\n<|im_end|>\n"
    messages = [system_segment, user_segment]

    full_output = ""
    collected_tokens: List[int] = []
    collected_entropies: List[float] = []
    event_windows: List[List[float]] = []
    pending_events = 0
    turn = 0

    generator_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    if hasattr(model.generation_config, "enable_thinking"):
        model.generation_config.enable_thinking = True
        generator_kwargs["enable_thinking"] = True
    else:
        print("[WARN] enable_thinking not supported by generation config; continuing without it.")

    stopping = KeywordStoppingCriteria(
        tokenizer=tokenizer,
        keywords=["</tool_call>", "</final_answer>"]
    )
    generator_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping])

    while turn < args.max_turns:
        turn += 1
        assistant_prefix = "<|im_start|>assistant\n"
        messages.append(assistant_prefix)
        input_ids, prompt_len = prepare_prompt(messages, tokenizer)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            output = model.generate(input_ids=input_ids, **generator_kwargs)

        new_tokens = output.sequences[0, prompt_len:]
        new_scores = output.scores or []
        if len(new_tokens) == 0:
            break

        entropies = [
            entropy_from_logits(score[0]).item()
            for score in new_scores
        ]

        collected_tokens.extend(new_tokens.tolist())
        collected_entropies.extend(entropies)

        decoded = decode_tokens(tokenizer, new_tokens)
        full_output += decoded
        messages[-1] = f"{assistant_prefix}{decoded}\n<|im_end|>\n"

        if pending_events > 0:
            # This assistant message responds to the previous tool call(s).
            event_windows.append(entropies)
            pending_events = 0

        tool_calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", decoded, re.S)
        for call in tool_calls:
            try:
                payload = json.loads(call)
                code = payload.get("code", "").strip()
            except json.JSONDecodeError:
                code = call.strip()
            result_json = run_python_tool(code, evaluator)
            tool_segment = (
                "<|im_start|>user\n"
                f"<tool_output>{result_json}</tool_output>\n"
                "<|im_end|>\n"
            )
            messages.append(tool_segment)
            full_output += f"<tool_output>{result_json}</tool_output>"
            pending_events += 1

        if "<final_answer>" in decoded:
            break

    final_answer = extract_final_answer(full_output) or ""
    is_correct = final_answer.strip() == format_dataset_answer(target_answer)

    return GenerationRecord(
        question=question,
        target_answer=format_dataset_answer(target_answer),
        model_answer=final_answer,
        full_output=full_output,
        entropies=collected_entropies,
        tokens=collected_tokens,
        event_windows=event_windows,
        is_correct=is_correct,
    )


def load_model_and_tokenizer(model_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def aggregate_entropy_windows(
    records: Sequence[GenerationRecord],
    window_size: int,
) -> List[np.ndarray]:
    curves: List[np.ndarray] = []
    for record in records:
        for window in record.event_windows:
            if not window:
                continue
            arr = np.array(window, dtype=np.float32)
            if arr.size >= window_size:
                arr = arr[:window_size]
            else:
                arr = np.pad(arr, (0, window_size - arr.size), constant_values=np.nan)
            curves.append(arr)
    return curves


def average_curve(curves: List[np.ndarray]) -> Optional[np.ndarray]:
    if not curves:
        return None
    stacked = np.stack(curves, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def run_model_experiment(
    model_name: str,
    model_path: str,
    dataset: Sequence[Dict[str, str]],
    device: torch.device,
    args: argparse.Namespace,
) -> List[GenerationRecord]:
    print(f"\n=== Running {model_name} ===")
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    model.eval()

    records: List[GenerationRecord] = []
    try:
        for idx, sample in enumerate(dataset):
            print(f"Processing sample {idx + 1}/{len(dataset)}...", end="\r")
            record = run_agent_on_sample(
                model=model,
                tokenizer=tokenizer,
                question=sample["question"],
                target_answer=sample["answer"],
                device=device,
                args=args,
            )
            records.append(record)
    finally:
        print()
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return records


def fabricate_dummy_records(num_samples: int, rng: random.Random) -> List[GenerationRecord]:
    records: List[GenerationRecord] = []
    for _ in range(num_samples):
        entropies = [rng.random() * 3 for _ in range(rng.randint(60, 120))]
        windows = []
        cursor = 0
        while cursor < len(entropies):
            length = rng.randint(10, 40)
            windows.append(entropies[cursor: cursor + length])
            cursor += length + rng.randint(5, 15)
        records.append(
            GenerationRecord(
                question="dummy",
                target_answer="42",
                model_answer="42",
                full_output="",
                entropies=entropies,
                tokens=[],
                event_windows=windows,
                is_correct=rng.random() > 0.5,
            )
        )
    return records


def split_records(records: Sequence[GenerationRecord]):
    correct = [rec for rec in records if rec.is_correct]
    incorrect = [rec for rec in records if not rec.is_correct]
    return correct, incorrect


def plot_entropy_curves(
    base_curves: Dict[str, Optional[np.ndarray]],
    arpo_curves: Dict[str, Optional[np.ndarray]],
    window_size: int,
    output_path: str,
) -> None:
    plt.figure(figsize=(10, 6))
    x = np.arange(window_size)

    styles = {
        "base_correct": ("Qwen2.5 Base - Correct", "tab:blue"),
        "base_incorrect": ("Qwen2.5 Base - Incorrect", "tab:orange"),
        "arpo_correct": ("Qwen2.5 ARPO - Correct", "tab:green"),
        "arpo_incorrect": ("Qwen2.5 ARPO - Incorrect", "tab:red"),
    }

    for key, (label, color) in styles.items():
        curve = base_curves.get(key) if key.startswith("base") else arpo_curves.get(key)
        if curve is None:
            continue
        plt.plot(x, curve, label=label, color=color)

    plt.xlabel("Tokens after tool feedback (t)")
    plt.ylabel("Average entropy (nats)")
    plt.title("Entropy trajectories after tool feedback events")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def summarize_group_statistics(name: str, records: Sequence[GenerationRecord]) -> str:
    entropies = [value for rec in records for value in rec.entropies]
    if not entropies:
        return f"{name}: no entropy records"
    return (
        f"{name}: {len(records)} trajectories, "
        f"mean entropy {statistics.mean(entropies):.3f}, "
        f"std {statistics.pstdev(entropies):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/Qwen2.5-7B")
    parser.add_argument("--arpo_model_path", type=str, default="/root/autodl-tmp/Qwen2.5-7B-ARPO")
    parser.add_argument("--dataset_size", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="evaluation")
    parser.add_argument("--plot_name", type=str, default="entropy_comparison.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Skip model calls and fabricate data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading GSM8K dataset (test split)...")
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = dataset.select(range(min(args.dataset_size, len(dataset))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        print("Running in dry-run mode: generating synthetic entropy records.")
        base_records = fabricate_dummy_records(args.dataset_size, random.Random(args.seed))
        arpo_records = fabricate_dummy_records(args.dataset_size, random.Random(args.seed + 1))
    else:
        base_records = run_model_experiment(
            model_name="Qwen2.5-7B",
            model_path=args.base_model_path,
            dataset=samples,
            device=device,
            args=args,
        )
        arpo_records = run_model_experiment(
            model_name="Qwen2.5-7B-ARPO",
            model_path=args.arpo_model_path,
            dataset=samples,
            device=device,
            args=args,
        )

    base_correct, base_incorrect = split_records(base_records)
    arpo_correct, arpo_incorrect = split_records(arpo_records)

    base_curves = {
        "base_correct": average_curve(aggregate_entropy_windows(base_correct, args.window_size)),
        "base_incorrect": average_curve(aggregate_entropy_windows(base_incorrect, args.window_size)),
    }
    arpo_curves = {
        "arpo_correct": average_curve(aggregate_entropy_windows(arpo_correct, args.window_size)),
        "arpo_incorrect": average_curve(aggregate_entropy_windows(arpo_incorrect, args.window_size)),
    }

    plot_path = os.path.join(args.output_dir, args.plot_name)
    plot_entropy_curves(base_curves, arpo_curves, args.window_size, plot_path)
    print(f"Saved entropy comparison plot to {plot_path}")

    print("\nGroup statistics:")
    for name, group in [
        ("Qwen2.5 Base - Correct", base_correct),
        ("Qwen2.5 Base - Incorrect", base_incorrect),
        ("Qwen2.5 ARPO - Correct", arpo_correct),
        ("Qwen2.5 ARPO - Incorrect", arpo_incorrect),
    ]:
        print(" - " + summarize_group_statistics(name, group))

    print("\nObservation Summary:")
    for label, curve in {**base_curves, **arpo_curves}.items():
        if curve is None:
            print(f" - {label}: insufficient tool-based events to compute curve.")
            continue
        trend = "decreasing" if np.all(np.diff(curve[:10]) < 0) else "mixed"
        print(
            f" - {label}: first 10 steps show {trend} trend, final entropy "
            f"{curve[min(len(curve) - 1, 99)]:.3f}"
        )


if __name__ == "__main__":
    main()
