import asyncio

import pytest

from utils.fulltext import (
    fulltext_field_from_index_path,
    fulltext_search_single_field,
    is_fulltext_field_path,
)


class Container:
    def __init__(self):
        self.queries = []

    def query_items(self, **kwargs):
        self.queries.append(kwargs["query"])

        async def rows():
            yield {"id": "one", "claims": ["fragrance free", "sensitive skin"]}

        return rows()


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ("/claims/[]", "claims[]"),
        ("/drug_active_ingredients/[]", "drug_active_ingredients[]"),
        ("/derived_attributes/[]", "derived_attributes[]"),
        ("/description", "description"),
        ("/description/?", "description"),
        ("/details/claims/[]", "details.claims[]"),
    ],
)
def test_policy_path_normalization(path, field):
    assert fulltext_field_from_index_path(path) == field
    assert is_fulltext_field_path(field)


@pytest.mark.parametrize("field", ["claims[]", "details.claims[]"])
def test_array_queries_filter_elements_without_scalar_ranking(field):
    container = Container()
    rows = asyncio.run(
        fulltext_search_single_field(
            container, field, "fragrance free sensitive skin", 15
        )
    )
    assert len(rows) == 1
    sql = container.queries[0]
    assert "WHERE EXISTS(" in sql
    assert f"element IN c.{field.removesuffix('[]')}" in sql
    assert 'FullTextContainsAny(element, "fragrance", "free", "sensitive", "skin")' in sql
    assert "FullTextScore" not in sql
    assert "ORDER BY" not in sql


def test_scalar_ranking_is_unchanged():
    container = Container()
    asyncio.run(fulltext_search_single_field(container, "product_title", "moisturizer", 15))
    assert container.queries == [
        'SELECT TOP 15 * FROM c ORDER BY RANK FullTextScore(c.product_title, "moisturizer")'
    ]


@pytest.mark.parametrize("field", ["claims[].text", "claims; DROP", "claims/[]", "claims[][]"])
def test_unsupported_fields_are_explicit_errors(field):
    assert not is_fulltext_field_path(field)
    with pytest.raises(ValueError, match="Invalid full-text field path"):
        asyncio.run(fulltext_search_single_field(Container(), field, "fragrance", 15))


def test_unsupported_policy_path_is_explicit_error():
    with pytest.raises(ValueError, match="Unsupported full-text index path"):
        fulltext_field_from_index_path("/claims/[]/nested")
