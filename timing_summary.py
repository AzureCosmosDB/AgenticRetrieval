import re
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "out"
TABLE_PATH = OUT_DIR / "timing_5q_compare_table.tsv"

TIMING_RE = re.compile(r"\[TIMING\]\s+(.*?):\s+\+([0-9]*\.?[0-9]+)s")
VEC_DONE_RE = re.compile(r"vector query – done \((\d+) results(?:,\s*([^)]+))?\)")
MAT_DONE_RE = re.compile(r"vector materialize x\d+ \(([^)]+)\) – done")


def find_timing_logs() -> list[Path]:
    logs = []
    for path in OUT_DIR.glob("timing_*.log"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "[TIMING]" in text:
            logs.append(path)
    logs.sort(key=lambda p: p.stat().st_mtime)
    return logs


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
    }

    for line in text.splitlines():
        match = TIMING_RE.search(line)
        if not match:
            continue

        label = match.group(1).strip()
        value = float(match.group(2))

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
        elif label.startswith("LLM preliminary – done"):
            data["llm_prelim"].append(value)
        elif label.startswith("LLM regenerate rnd 1 – done"):
            data["llm_regen1"].append(value)
        elif label.startswith("LLM gap-decompose – done"):
            data["llm_gap"].append(value)
        elif label.startswith("pipeline.run – TOTAL"):
            data["pipeline_total"].append(value)

    err_lines = [line for line in text.splitlines() if line.startswith("Error:")]
    badrequest = text.count("BadRequestError on")
    max_retry = text.count("Max retries exceeded")

    data["_meta"] = {
        "errors": len(err_lines),
        "badrequest": badrequest,
        "max_retry": max_retry,
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


def pct_change(prev, curr):
    if prev is None or curr is None or prev == 0:
        return "NA"
    return f"{((curr - prev) / prev) * 100:+.1f}%"


def range_width(rng):
    if rng is None:
        return None
    return rng[1] - rng[0]


def pct_change_range(prev_rng, curr_rng):
    prev_w = range_width(prev_rng)
    curr_w = range_width(curr_rng)
    if prev_w is None or curr_w is None or prev_w == 0:
        return "NA"
    return f"{((curr_w - prev_w) / prev_w) * 100:+.1f}%"


def build_metric_values(data):
    values = {}
    values["retrieve() TOTAL – single call"] = fmt_single(first_wave_mean(data["retrieve_total"], 5))
    values["retrieve() TOTAL – under sub-Q contention"] = fmt_range(contention_range(data["retrieve_total"], 5))
    values["Fulltext query – single"] = fmt_single(first_wave_mean(data["fulltext_done"], 5))
    values["Fulltext query – under contention (sub-Q parallel)"] = fmt_range(contention_range(data["fulltext_done"], 5))
    values["Vector query – single (all sources)"] = fmt_single(first_wave_mean(data["vector_done"], 5))
    values["Vector query – under contention (all sources)"] = fmt_range(contention_range(data["vector_done"], 5))
    values["Vector materialize/read stage (all sources)"] = fmt_range(full_range(data["mat_done"]))

    for container in sorted(data["vector_by_container"]):
        vals = data["vector_by_container"][container]
        values[f"Vector query – single ({container})"] = fmt_single(first_wave_mean(vals, 5))
        values[f"Vector query – under contention ({container})"] = fmt_range(contention_range(vals, 5))

    for container in sorted(data["mat_by_container"]):
        vals = data["mat_by_container"][container]
        values[f"Vector materialize/read stage ({container})"] = fmt_range(full_range(vals))

    values["LLM sub-Q answer"] = fmt_range(full_range(data["llm_subq"]))
    values["LLM synthesis"] = fmt_single(mean(data["llm_synth"]))
    values["LLM preliminary"] = fmt_single(mean(data["llm_prelim"]))
    values["LLM regenerate rnd 1"] = fmt_single(mean(data["llm_regen1"]))
    values["LLM gap-decompose"] = fmt_range(full_range(data["llm_gap"]))
    values["pipeline.run TOTAL"] = fmt_single(mean(data["pipeline_total"]))
    return values


def parse_single_seconds(value):
    if not value or value == "NA":
        return None
    if "–" in value or "-" in value:
        return None
    try:
        return float(value.rstrip("s"))
    except ValueError:
        return None


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


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


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
            "No timing logs found in out/. Run rag_divdet.py with --timing and capture output to a .log file in out/."
        )

    current_log_path = timing_logs[-1]
    current_log_text = current_log_path.read_text(encoding="utf-8", errors="ignore")

    previous_log_path = None
    previous_log_text = None
    for candidate in reversed(timing_logs[:-1]):
        candidate_text = candidate.read_text(encoding="utf-8", errors="ignore")
        if candidate_text != current_log_text:
            previous_log_path = candidate
            previous_log_text = candidate_text
            break

    current_data = parse_timings(current_log_text)
    current_values = build_metric_values(current_data)
    ordered_components = list(current_values.keys())

    lines = []
    if previous_log_text is None:
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
        previous_data = parse_timings(previous_log_text)
        previous_values = build_metric_values(previous_data)

        lines.append(f"Prev log\t{previous_log_path.name}")
        lines.append(f"This log\t{current_log_path.name}")
        lines.append("")
        lines.append("Component\tPrev run\tThis run\tChange")
        for component in ordered_components:
            prev_value = previous_values.get(component, "NA")
            curr_value = current_values.get(component, "NA")

            prev_range = parse_range_seconds(prev_value)
            curr_range = parse_range_seconds(curr_value)

            if prev_range is not None and curr_range is not None:
                change = pct_change_range(prev_range, curr_range)
            else:
                change = pct_change(parse_single_seconds(prev_value), parse_single_seconds(curr_value))
            lines.append(f"{component}\t{prev_value}\t{curr_value}\t{change}")

        lines.append("")
        lines.append(
            f"_meta prev_errors={previous_data['_meta']['errors']} new_badrequest_errors={current_data['_meta']['badrequest']} new_max_retry_exceeded={current_data['_meta']['max_retry']}"
        )

    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")

    if previous_log_path is None:
        print(f"Log: {current_log_path.name}")
    else:
        print(f"Prev log: {previous_log_path.name}")
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
