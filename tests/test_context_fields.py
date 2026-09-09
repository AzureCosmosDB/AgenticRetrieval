from dynamic_retriever import (
    _build_tool_use_query_template,
    _build_traceability_reminder,
)
from prompts import DEFAULT_QUERY_TEMPLATE


def test_tool_use_planning_prompt_is_unaffected_by_context_fields():
    prompt = _build_tool_use_query_template(
        DEFAULT_QUERY_TEMPLATE,
        prune_k=20,
    )

    assert "context fields" not in prompt.lower()
    assert "Question: {question}" in prompt


def test_traceability_reminder_names_configured_fields():
    reminder = _build_traceability_reminder(
        ["asset_ref", "region_code", "asset_ref"]
    )

    assert "asset_ref, region_code" in reminder
    assert "Copy each field/value pair exactly" in reminder
    assert "measurements, claims, categories, summaries" in reminder


def test_traceability_reminder_is_empty_without_configured_fields():
    assert _build_traceability_reminder([]) == ""
