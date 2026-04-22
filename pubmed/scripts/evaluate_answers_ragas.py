"""Compare two answer JSON files using RAGAS metrics.

Evaluates each answer set independently on multiple criteria, then prints
a side-by-side comparison. Uses Azure OpenAI via litellm + Azure AD token.

Usage:
    python evaluate_answers_ragas.py \
        --base output/review_answers_64.json \
        --other output/review_answers_divdet_64_128_rp5_eta01.json \
        --base-name vector_64 \
        --other-name divdet_64 \
        --output output/eval_ragas_vector_64_vs_divdet_64.json
"""

import argparse
import asyncio
import json
import logging
import sys
import time

import litellm
import yaml
from azure.identity import AzureCliCredential
from ragas.llms import llm_factory
from ragas.metrics import NumericMetric
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Suppress noisy loggers
for name in ["azure", "httpx", "openai", "litellm", "instructor", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Metrics definitions
# ---------------------------------------------------------------------------
METRICS = {
    "comprehensiveness": NumericMetric(
        name="comprehensiveness",
        prompt=(
            "Rate how comprehensively the answer covers the question on a scale "
            "of 0.0 to 1.0. Consider breadth of coverage, depth of detail, and "
            "whether important aspects are missing.\n\n"
            "Question: {question}\nAnswer: {answer}"
        ),
    ),
    "evidence_use": NumericMetric(
        name="evidence_use",
        prompt=(
            "Rate how well the answer uses evidence from retrieved sources on a "
            "scale of 0.0 to 1.0. Consider citation of specific papers, use of "
            "concrete data/findings, and grounding of claims.\n\n"
            "Question: {question}\nAnswer: {answer}"
        ),
    ),
    "accuracy": NumericMetric(
        name="accuracy",
        prompt=(
            "Rate the factual accuracy and scientific rigor of the answer on a "
            "scale of 0.0 to 1.0. Consider whether claims are well-supported, "
            "nuanced, and free of contradictions or errors.\n\n"
            "Question: {question}\nAnswer: {answer}"
        ),
    ),
    "clarity": NumericMetric(
        name="clarity",
        prompt=(
            "Rate the clarity and organization of the answer on a scale of 0.0 "
            "to 1.0. Consider logical structure, readability, effective use of "
            "headings/sections, and overall coherence.\n\n"
            "Question: {question}\nAnswer: {answer}"
        ),
    ),
    "relevance": NumericMetric(
        name="relevance",
        prompt=(
            "Rate how relevant the answer is to the specific question asked on "
            "a scale of 0.0 to 1.0. Consider whether it directly addresses the "
            "question without excessive tangential content.\n\n"
            "Question: {question}\nAnswer: {answer}"
        ),
    ),
}


def _make_llm(cred, endpoint, deployment, api_version, token_scope):
    """Create a fresh ragas LLM with a new token."""
    token = cred.get_token(token_scope).token
    return llm_factory(
        f"azure/{deployment}",
        provider="litellm",
        client=litellm.completion,
        api_base=endpoint,
        api_version=api_version,
        api_key=token,
    )


def build_llm(llm_cfg):
    """Create a ragas LLM using Azure AD token via litellm."""
    endpoint = llm_cfg.get("endpoint", "")
    deployment = llm_cfg.get("model", "gpt-5.4")
    api_version = llm_cfg.get("api_version", "2024-12-01-preview")
    token_scope = llm_cfg.get("token_scope", "https://cognitiveservices.azure.com/.default")
    cred = AzureCliCredential()
    llm = _make_llm(cred, endpoint, deployment, api_version, token_scope)
    return llm, cred, {"endpoint": endpoint, "deployment": deployment,
                       "api_version": api_version, "token_scope": token_scope}


def load_answers(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_answers(answers: list[dict], llm, cred, llm_params: dict, name: str) -> list[dict]:
    """Score each answer on all metrics. Returns list of per-question result dicts."""
    results = []
    for i, item in enumerate(tqdm(answers, desc=f"Scoring [{name}]")):
        question = item.get("question", "")
        answer = item.get("answer", "")
        row = {"question_idx": i, "question": question[:120]}

        # Rebuild LLM with fresh token every 10 questions to avoid expiry
        if i % 10 == 0 and i > 0:
            llm = _make_llm(cred, **llm_params)
            log.info("Rebuilt LLM with fresh token")

        for metric_name, metric in METRICS.items():
            for attempt in range(5):
                try:
                    result = metric.score(llm=llm, question=question, answer=answer)
                    row[metric_name] = result.value
                    break
                except Exception as e:
                    err_str = str(e)
                    if "Authentication" in err_str or "401" in err_str or "403" in err_str:
                        log.warning(f"[{name}] q{i} {metric_name} auth error, rebuilding LLM...")
                        llm = _make_llm(cred, **llm_params)
                    if attempt < 4:
                        wait = min(5.0 * (2 ** attempt), 60)
                        log.warning(f"[{name}] q{i} {metric_name} attempt {attempt+1} failed: {err_str[:100]}. Retrying in {wait:.0f}s...")
                        time.sleep(wait)
                    else:
                        log.error(f"[{name}] q{i} {metric_name} failed after 5 attempts: {e}")
                        row[metric_name] = None

        results.append(row)
        if (i + 1) % 10 == 0:
            log.info(f"[{name}] Scored {i+1}/{len(answers)} questions")

    log.info(f"[{name}] Done scoring {len(answers)} questions")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare two answer files using RAGAS metrics")
    parser.add_argument("--config", default="config.pubmed.yaml", help="Path to YAML config file")
    parser.add_argument("--base", required=True, help="Base answer JSON file")
    parser.add_argument("--other", required=True, help="Other answer JSON file")
    parser.add_argument("--base-name", default="base", help="Name for base method")
    parser.add_argument("--other-name", default="other", help="Name for other method")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg.get("llm", {})

    base_answers = load_answers(args.base)
    other_answers = load_answers(args.other)
    log.info(f"Loaded {len(base_answers)} from {args.base}, {len(other_answers)} from {args.other}")

    assert len(base_answers) == len(other_answers), "Answer files must have the same number of questions"

    llm, cred, llm_params = build_llm(llm_cfg)

    log.info(f"Evaluating '{args.base_name}'...")
    base_results = evaluate_answers(base_answers, llm, cred, llm_params, args.base_name)

    log.info(f"Evaluating '{args.other_name}'...")
    other_results = evaluate_answers(other_answers, llm, cred, llm_params, args.other_name)

    # Combine results
    combined = []
    for b, o in zip(base_results, other_results):
        row = {"question_idx": b["question_idx"], "question": b["question"]}
        for metric_name in METRICS:
            row[f"{args.base_name}_{metric_name}"] = b.get(metric_name)
            row[f"{args.other_name}_{metric_name}"] = o.get(metric_name)
        combined.append(row)

    # Save
    output_path = args.output or f"eval_ragas_{args.base_name}_vs_{args.other_name}.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    log.info(f"Saved results to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"RAGAS Comparison: {args.base_name} vs {args.other_name}")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {args.base_name:>15} {args.other_name:>15} {'delta':>10}")
    print(f"{'-'*70}")
    for metric_name in METRICS:
        base_vals = [r.get(f"{args.base_name}_{metric_name}") for r in combined if r.get(f"{args.base_name}_{metric_name}") is not None]
        other_vals = [r.get(f"{args.other_name}_{metric_name}") for r in combined if r.get(f"{args.other_name}_{metric_name}") is not None]
        base_mean = sum(base_vals) / len(base_vals) if base_vals else 0
        other_mean = sum(other_vals) / len(other_vals) if other_vals else 0
        delta = other_mean - base_mean
        sign = "+" if delta >= 0 else ""
        print(f"{metric_name:<25} {base_mean:>15.4f} {other_mean:>15.4f} {sign}{delta:>9.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
