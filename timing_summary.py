import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "out"
TABLE_PATH = OUT_DIR / "timing_5q_compare_table.tsv"

TIMING_ENTRY_RE = re.compile(
    r"(?:\[TIMING\]|¤)\s+"
    r"(?P<body>.+?)(?=:\s+\+[0-9]*\.?[0-9]+s\s+\(total\s+[0-9]*\.?[0-9]+s\))"
    r":\s+\+(?P<value>[0-9]*\.?[0-9]+)s\s+\(total\s+(?P<total>[0-9]*\.?[0-9]+)s\)"
)
PROCESSING_RE = re.compile(r"Processing\s+(\d+)\s+questions")
VEC_DONE_RE = re.compile(r"vector query – done \((\d+) results(?:,\s*([^)]+))?\)")
MAT_DONE_RE = re.compile(r"vector materialize x\d+ \(([^)]+)\) – done")
TOTAL_PROMPT_TOKENS_RE = re.compile(
    r"Total premium prompt tokens:\s+(?P<tokens>[\d,]+)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def find_timing_logs() -> list[Path]:
    logs = []
    for path in OUT_DIR.glob("timing_*.log"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "[TIMING]" in text or "¤" in text:
            logs.append(path)
    logs.sort(key=lambda p: p.stat().st_mtime)
    return logs


def is_completed_timing_log(text: str) -> bool:
    clean_text = strip_ansi(text)
    return ("pipeline.run – TOTAL" in clean_text) or ("pipeline.run_efficient – TOTAL" in clean_text)


def parse_timings(text: str):
    data = {
        "retrieve_total": [],
        "fulltext_done": [],
        "vector_done": [],
        "mat_done": [],
        "vector_by_container": {},
        "mat_by_container": {},
        "llm_subq": [],
        "llm_synth": [],
        "llm_prelim": [],
        "llm_regen1": [],
        "llm_gap": [],
        "pipeline_total": [],
        "run_totals": [],
    }

    clean_text = strip_ansi(text)

    def normalize_label(body: str) -> str:
        label = body.strip()
        if ": " in label:
            prefix, rest = label.split(": ", 1)
            valid_label_prefixes = (
                "pipeline.",
                "pipeline:",
                "retrieve",
                "fulltext",
                "vector",
                "LLM ",
                "embed",
            )
            if rest.startswith(valid_label_prefixes):
                return rest.strip()
        return label

    for match in TIMING_ENTRY_RE.finditer(clean_text):
        label = normalize_label(match.group("body"))
        value = float(match.group("value"))
        data["run_totals"].append(float(match.group("total")))

        if label.startswith("retrieve – TOTAL"):
            data["retrieve_total"].append(value)
        elif label.startswith("fulltext query – done"):
            data["fulltext_done"].append(value)
        elif label.startswith("vector query – done"):
            vm = VEC_DONE_RE.search(label)
            if vm:
                data["vector_done"].append(value)
                container = (vm.group(2) or "unknown").strip()
                data["vector_by_container"].setdefault(container, []).append(value)
        elif label.startswith("vector materialize"):
            data["mat_done"].append(value)
            mm = MAT_DONE_RE.search(label)
            if mm:
                container = (mm.group(1) or "unknown").strip()
                data["mat_by_container"].setdefault(container, []).append(value)
        elif label.startswith("LLM sub-Q answer – done"):
            data["llm_subq"].append(value)
        elif label.startswith("LLM synthesis – done"):
            data["llm_synth"].append(value)
        elif label.startswith("LLM efficient synthesis – done"):
            data["llm_synth"].append(value)
        elif label.startswith("LLM preliminary – done"):
            data["llm_prelim"].append(value)
        elif label.startswith("LLM regenerate rnd 1 – done"):
            data["llm_regen1"].append(value)
        elif label.startswith("LLM efficient regen rnd 1 – done"):
            data["llm_regen1"].append(value)
        elif label.startswith("LLM gap-decompose – done"):
            data["llm_gap"].append(value)
        elif label.startswith("pipeline.run – TOTAL"):
            data["pipeline_total"].append(value)
        elif label.startswith("pipeline.run_efficient – TOTAL"):
            data["pipeline_total"].append(value)

    err_lines = [line for line in clean_text.splitlines() if line.startswith("Error:")]
    badrequest = clean_text.count("BadRequestError on")
    max_retry = clean_text.count("Max retries exceeded")
    processing_match = PROCESSING_RE.search(clean_text)
    questions = int(processing_match.group(1)) if processing_match else None
    total_prompt_tokens_match = TOTAL_PROMPT_TOKENS_RE.search(clean_text)
    total_prompt_tokens = (
        int(total_prompt_tokens_match.group("tokens").replace(",", ""))
        if total_prompt_tokens_match
        else None
    )
    run_wall_total = max(data["run_totals"]) if data["run_totals"] else None
    wall_per_question = (
        (run_wall_total / questions)
        if (run_wall_total is not None and questions and questions > 0)
        else None
    )
    prompt_tokens_per_question = (
        (total_prompt_tokens / questions)
        if (total_prompt_tokens is not None and questions and questions > 0)
        else None
    )

    data["_meta"] = {
        "errors": len(err_lines),
        "badrequest": badrequest,
        "max_retry": max_retry,
        "questions": questions,
        "total_prompt_tokens": total_prompt_tokens,
        "prompt_tokens_per_question": prompt_tokens_per_question,
        "run_wall_total": run_wall_total,
        "run_wall_per_question": wall_per_question,
    }
    return data


def mean(values):
    return sum(values) / len(values) if values else None


def first_wave_mean(values, n=5):
    return mean(values[:n]) if values else None


def contention_range(values, n=5):
    tail = values[n:] if len(values) > n else []
    if not tail:
        return None
    return min(tail), max(tail)


def full_range(values):
    if not values:
        return None
    return min(values), max(values)


def fmt_single(value):
    return "NA" if value is None else f"{value:.2f}s"


def fmt_range(rng):
    return "NA" if rng is None else f"{rng[0]:.2f}–{rng[1]:.2f}s"


def fmt_tokens(value):
    return "NA" if value is None else f"{value:,.0f}"


def fmt_tokens_per_question(value):
    return "NA" if value is None else f"{value:,.0f} tokens/q"


def pct_change(prev, curr):
    if prev is None or curr is None or prev == 0:
        return "NA"
    return f"{((curr - prev) / prev) * 100:+.1f}%"


def range_width(rng):
    if rng is None:
        return None
    return rng[1] - rng[0]


def pct_change_range(prev_rng, curr_rng):
    prev_value = mean(prev_rng)
    curr_value = mean(curr_rng)
    if prev_value is None or curr_value is None or prev_value == 0:
        return "NA"
    return f"{((curr_value - prev_value) / prev_value) * 100:+.1f}%"


def build_metric_values(data):
    values = {}
    values["retrieve() TOTAL – single call"] = fmt_single(first_wave_mean(data["retrieve_total"], 5))
    values["retrieve() TOTAL – under sub-Q contention"] = fmt_range(contention_range(data["retrieve_total"], 5))
    values["Fulltext query – single"] = fmt_single(first_wave_mean(data["fulltext_done"], 5))
    values["Fulltext query – under contention (sub-Q parallel)"] = fmt_range(contention_range(data["fulltext_done"], 5))
    values["Vector query – single (all sources)"] = fmt_single(first_wave_mean(data["vector_done"], 5))
    values["Vector query – under contention (all sources)"] = fmt_range(contention_range(data["vector_done"], 5))

    for container in sorted(data["vector_by_container"]):
        vals = data["vector_by_container"][container]
        values[f"Vector query – single ({container})"] = fmt_single(first_wave_mean(vals, 5))
        values[f"Vector query – under contention ({container})"] = fmt_range(contention_range(vals, 5))

    values["LLM synthesis"] = fmt_single(mean(data["llm_synth"]))
    values["LLM preliminary"] = fmt_single(mean(data["llm_prelim"]))
    values["LLM regenerate rnd 1"] = fmt_single(mean(data["llm_regen1"]))
    values["LLM gap-decompose"] = fmt_range(full_range(data["llm_gap"]))
    questions = data.get("_meta", {}).get("questions")
    question_note = f" [{questions} questions]" if isinstance(questions, int) and questions > 0 else ""
    values[f"premium prompt tokens TOTAL{question_note}"] = fmt_tokens(
        data.get("_meta", {}).get("total_prompt_tokens")
    )
    values["premium prompt tokens / question"] = fmt_tokens_per_question(
        data.get("_meta", {}).get("prompt_tokens_per_question")
    )
    values["run wall TOTAL / question"] = fmt_single(data.get("_meta", {}).get("run_wall_per_question"))
    values[f"pipeline.run TOTAL{question_note}"] = fmt_single(mean(data["pipeline_total"]))
    return values


def parse_single_seconds(value):
    if not value or value == "NA":
        return None
    if "–" in value or "-" in value:
        return None
    match = re.search(r"([0-9][0-9,]*\.?[0-9]*)", value)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def parse_range_seconds(value):
    if not value or value == "NA":
        return None
    if "–" in value:
        left, right = value.rstrip("s").split("–", 1)
    elif "-" in value:
        left, right = value.rstrip("s").split("-", 1)
    else:
        return None
    try:
        return float(left), float(right)
    except ValueError:
        return None


def supports_color() -> bool:
    return sys.stdout.isatty()


def colorize(text: str, code: str) -> str:
    if not supports_color():
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def parse_tsv_lines(lines):
    header_index = next((i for i, line in enumerate(lines) if line.startswith("Component\t")), None)
    if header_index is None:
        return []

    rows = []
    for line in lines[header_index:]:
        if not line:
            continue
        if line.startswith("_meta"):
            continue
        rows.append(line.split("\t"))
    return rows


def find_previous_runs(current_log_text, completed_logs, timing_logs, limit=2):
    previous_runs = []
    seen_texts = {current_log_text}
    candidate_pool = (
        completed_logs[:-1]
        if completed_logs
        else [(path, path.read_text(encoding="utf-8", errors="ignore")) for path in timing_logs[:-1]]
    )

    for candidate_path, candidate_text in reversed(candidate_pool):
        if candidate_text in seen_texts:
            continue
        previous_runs.append((candidate_path, candidate_text))
        seen_texts.add(candidate_text)
        if len(previous_runs) == limit:
            break

    previous_runs.reverse()
    return previous_runs


def render_pretty_table(lines):
    rows = parse_tsv_lines(lines)
    if not rows:
        return ""

    col_count = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (col_count - len(row)) for row in rows]

    widths = [0] * col_count
    for row in normalized_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(strip_ansi(cell)))

    h = "┄"
    v = "┆"
    top = "┌" + "┬".join(h * (w + 2) for w in widths) + "┐"
    mid = "├" + "┼".join(h * (w + 2) for w in widths) + "┤"
    bottom = "└" + "┴".join(h * (w + 2) for w in widths) + "┘"

    def color_fg_only(text: str, code: str) -> str:
        if not supports_color():
            return text
        return f"\x1b[{code}m{text}\x1b[22;39m"

    out = [top]
    for row_idx, row in enumerate(normalized_rows):
        is_zebra = row_idx > 0 and row_idx % 2 == 0
        row_prefix = "\x1b[48;5;236m" if supports_color() and is_zebra else ""
        row_suffix = "\x1b[0m" if supports_color() and is_zebra else ""

        rendered_cells = []
        for col_idx, cell in enumerate(row):
            text = cell.ljust(widths[col_idx])
            if row_idx == 0:
                text = color_fg_only(text, "1;36")
            elif col_idx == 0:
                text = color_fg_only(text, "1;33")

            rendered_cells.append(f" {text} ")

        out.append(f"{row_prefix}{v}{v.join(rendered_cells)}{v}{row_suffix}")
        if row_idx == 0:
            out.append(mid)

    out.append(bottom)
    return "\n".join(out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timing_logs = find_timing_logs()
    if not timing_logs:
        raise RuntimeError(
            "No timing logs found in out/. Run agentic_retriever.py with --timing and capture output to a .log file in out/."
        )

    completed_logs: list[tuple[Path, str]] = []
    for path in timing_logs:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if is_completed_timing_log(text):
            completed_logs.append((path, text))

    if completed_logs:
        current_log_path, current_log_text = completed_logs[-1]
    else:
        current_log_path = timing_logs[-1]
        current_log_text = current_log_path.read_text(encoding="utf-8", errors="ignore")

    previous_runs = find_previous_runs(current_log_text, completed_logs, timing_logs, limit=2)

    current_data = parse_timings(current_log_text)
    current_values = build_metric_values(current_data)
    ordered_components = list(current_values.keys())

    lines = []
    if not previous_runs:
        lines.append(f"Log\t{current_log_path.name}")
        lines.append("")
        lines.append("Component\tThis run")
        for component in ordered_components:
            lines.append(f"{component}\t{current_values[component]}")
        lines.append("")
        lines.append(
            f"_meta this_badrequest_errors={current_data['_meta']['badrequest']} this_max_retry_exceeded={current_data['_meta']['max_retry']}"
        )
    else:
        previous_summaries = []
        for previous_log_path, previous_log_text in previous_runs:
            previous_data = parse_timings(previous_log_text)
            previous_values = build_metric_values(previous_data)
            previous_summaries.append((previous_log_path, previous_data, previous_values))

        for index, (previous_log_path, _, _) in enumerate(previous_summaries, start=1):
            lines.append(f"Prev log {index}\t{previous_log_path.name}")
        lines.append(f"This log\t{current_log_path.name}")
        lines.append("")

        header = ["Component"]
        header.extend(f"Prev run {index}" for index in range(1, len(previous_summaries) + 1))
        header.extend(["This run", "Change"])
        lines.append("\t".join(header))

        latest_previous_values = previous_summaries[-1][2]
        for component in ordered_components:
            previous_values_for_component = [summary[2].get(component, "NA") for summary in previous_summaries]
            curr_value = current_values.get(component, "NA")
            prev_value = latest_previous_values.get(component, "NA")

            prev_range = parse_range_seconds(prev_value)
            curr_range = parse_range_seconds(curr_value)

            if prev_range is not None and curr_range is not None:
                change = pct_change_range(prev_range, curr_range)
            else:
                change = pct_change(parse_single_seconds(prev_value), parse_single_seconds(curr_value))
            row = [component, *previous_values_for_component, curr_value, change]
            lines.append("\t".join(row))

        lines.append("")
        meta_parts = []
        for index, (_, previous_data, _) in enumerate(previous_summaries, start=1):
            meta_parts.append(f"prev{index}_errors={previous_data['_meta']['errors']}")
        meta_parts.append(f"new_badrequest_errors={current_data['_meta']['badrequest']}")
        meta_parts.append(f"new_max_retry_exceeded={current_data['_meta']['max_retry']}")
        lines.append("_meta " + " ".join(meta_parts))

    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")

    if not previous_runs:
        print(f"Log: {current_log_path.name}")
    else:
        for index, (previous_log_path, _) in enumerate(previous_runs, start=1):
            print(f"Prev log {index}: {previous_log_path.name}")
        print(f"This log: {current_log_path.name}")
    print()

    print(render_pretty_table(lines))

    meta_lines = [line for line in lines if line.startswith("_meta")]
    if meta_lines:
        print()
        for meta_line in meta_lines:
            print(meta_line)


if __name__ == "__main__":
    main()
