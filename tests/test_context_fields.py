import pytest

import dynamic_retriever as dr
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


def test_tool_use_format_prioritizes_context_fields_before_long_content(monkeypatch):
    monkeypatch.setattr(dr, "_tool_use_context_fields", ["upc", "product_id", "upc", "missing"])
    monkeypatch.setattr(dr, "_tool_use_all_embed", {"vector"})
    description = "Long product description. " * 100
    doc = {
        "description": description,
        "product_id": 0,
        "upc": "001234567890",
        "title": "Product",
        "_etag": "hidden",
        "vector": [1.0],
        "e": [1.0],
    }

    text = dr.tool_use_fmt(doc)

    assert text == (
        f"upc: 001234567890\nproduct_id: 0\ndescription: {description}\ntitle: Product"
    )
    assert "upc: 001234567890" in text[:512]
    assert list(doc)[0] == "description"


@pytest.mark.parametrize("value", [0, False, "", [], {}, None])
def test_tool_use_format_preserves_context_field_value_filtering(monkeypatch, value):
    monkeypatch.setattr(dr, "_tool_use_context_fields", ["reference"])
    monkeypatch.setattr(dr, "_tool_use_all_embed", set())

    text = dr.tool_use_fmt({"other": value, "reference": value})

    assert text == ("" if value is None else f"reference: {value}")


def test_tool_use_format_without_context_fields_preserves_document_order(monkeypatch):
    monkeypatch.setattr(dr, "_tool_use_context_fields", [])
    monkeypatch.setattr(dr, "_tool_use_all_embed", set())

    assert dr.tool_use_fmt({"description": "Product", "upc": "001"}) == (
        "description: Product\nupc: 001"
    )
