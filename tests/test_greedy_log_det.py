"""Tests for greedy_log_det_select."""

import numpy as np
import pytest

from greedy_log_det import greedy_log_det_select


def _make_rng(seed: int = 42):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------

def test_returns_k_indices():
    """Should return exactly k indices when k < n."""
    rng = _make_rng()
    vectors = rng.standard_normal((10, 4)).astype(np.float32)
    query = rng.standard_normal(4).astype(np.float32)
    result = greedy_log_det_select(vectors, query, k=3)
    assert len(result) == 3


def test_indices_in_range():
    """All returned indices must be valid row indices."""
    rng = _make_rng()
    n, d = 8, 6
    vectors = rng.standard_normal((n, d)).astype(np.float32)
    query = rng.standard_normal(d).astype(np.float32)
    result = greedy_log_det_select(vectors, query, k=4)
    assert all(0 <= i < n for i in result)


def test_no_duplicate_indices():
    """Returned indices must be unique."""
    rng = _make_rng()
    vectors = rng.standard_normal((10, 5)).astype(np.float32)
    query = rng.standard_normal(5).astype(np.float32)
    result = greedy_log_det_select(vectors, query, k=5)
    assert len(result) == len(set(result))


def test_k_ge_n_returns_all():
    """When k >= n the function should return all indices."""
    rng = _make_rng()
    n, d = 5, 4
    vectors = rng.standard_normal((n, d)).astype(np.float32)
    query = rng.standard_normal(d).astype(np.float32)
    result = greedy_log_det_select(vectors, query, k=n)
    assert sorted(result) == list(range(n))
    result_overflow = greedy_log_det_select(vectors, query, k=n + 3)
    assert sorted(result_overflow) == list(range(n))


def test_k_one_returns_single_index():
    """k=1 should return a list with exactly one valid index."""
    rng = _make_rng()
    vectors = rng.standard_normal((6, 3)).astype(np.float32)
    query = rng.standard_normal(3).astype(np.float32)
    result = greedy_log_det_select(vectors, query, k=1)
    assert len(result) == 1
    assert 0 <= result[0] < 6


# ---------------------------------------------------------------------------
# eta=0 vs eta>0 paths
# ---------------------------------------------------------------------------

def test_eta_zero_path():
    """eta=0 (Gram–Schmidt path) should produce valid output."""
    rng = _make_rng(0)
    vectors = rng.standard_normal((12, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)
    result = greedy_log_det_select(vectors, query, k=5, eta=0.0)
    assert len(result) == 5
    assert len(set(result)) == 5


def test_eta_positive_path():
    """eta>0 (Woodbury path) should produce valid output."""
    rng = _make_rng(1)
    vectors = rng.standard_normal((12, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)
    result = greedy_log_det_select(vectors, query, k=5, eta=1.0)
    assert len(result) == 5
    assert len(set(result)) == 5


# ---------------------------------------------------------------------------
# rescale_power tests
# ---------------------------------------------------------------------------

def test_rescale_power_zero_no_change():
    """rescale_power=0 (default) should not change the vectors used."""
    rng = _make_rng(7)
    vectors = rng.standard_normal((10, 4)).astype(np.float32)
    query = rng.standard_normal(4).astype(np.float32)
    # Run twice – must be deterministic with same seed inputs
    r1 = greedy_log_det_select(vectors, query, k=3, rescale_power=0.0)
    r2 = greedy_log_det_select(vectors, query, k=3, rescale_power=0.0)
    assert r1 == r2


def test_rescale_power_positive():
    """rescale_power>0 should still return valid k distinct indices."""
    rng = _make_rng(3)
    vectors = np.abs(rng.standard_normal((10, 4))).astype(np.float32)
    query = np.abs(rng.standard_normal(4)).astype(np.float32)
    result = greedy_log_det_select(vectors, query, k=4, rescale_power=0.5)
    assert len(result) == 4
    assert len(set(result)) == 4


# ---------------------------------------------------------------------------
# Diversity property: orthogonal vectors should all be selected first
# ---------------------------------------------------------------------------

def test_orthogonal_vectors_selected():
    """With orthogonal basis vectors + noise, basis vectors should be selected."""
    d = 4
    # Perfect orthonormal basis
    basis = np.eye(d, dtype=np.float32)
    # Add low-norm noise vectors
    rng = _make_rng(99)
    noise = (rng.standard_normal((6, d)) * 0.01).astype(np.float32)
    vectors = np.vstack([basis, noise])  # shape (10, 4)
    query = np.ones(d, dtype=np.float32) / np.sqrt(d)
    result = greedy_log_det_select(vectors, query, k=d)
    # The four basis vectors (indices 0–3) should be chosen
    assert set(result) == {0, 1, 2, 3}


def test_orthogonal_vectors_selected_eta_positive():
    """Same as above but with eta>0 (Woodbury path)."""
    d = 4
    basis = np.eye(d, dtype=np.float32)
    rng = _make_rng(100)
    noise = (rng.standard_normal((6, d)) * 0.01).astype(np.float32)
    vectors = np.vstack([basis, noise])
    query = np.ones(d, dtype=np.float32) / np.sqrt(d)
    result = greedy_log_det_select(vectors, query, k=d, eta=1e-4)
    assert set(result) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Fewer-than-k returns when vectors are nearly linearly dependent
# ---------------------------------------------------------------------------

def test_fewer_than_k_linearly_dependent_eta_zero():
    """eta=0: when all vectors are identical, only one can be selected."""
    d = 4
    v = np.ones(d, dtype=np.float32) / np.sqrt(d)
    vectors = np.tile(v, (6, 1))
    query = v.copy()
    result = greedy_log_det_select(vectors, query, k=4)
    assert len(result) < 4
    assert len(result) >= 1


def test_fewer_than_k_linearly_dependent_eta_positive():
    """eta>0: nearly linearly dependent vectors may trigger early stop."""
    d = 4
    v = np.ones(d, dtype=np.float32) / np.sqrt(d)
    vectors = np.tile(v, (6, 1))
    query = v.copy()
    result = greedy_log_det_select(vectors, query, k=4, eta=1e-30)
    assert len(result) >= 1
    assert len(result) <= 4


# ---------------------------------------------------------------------------
# sqrt NaN guard: negative scores from float roundoff must not produce NaN
# ---------------------------------------------------------------------------

def test_no_nan_from_float_roundoff():
    """r_norm must not be NaN even when scores[best_i] is tiny negative."""
    d = 3
    # Construct vectors where Gram-Schmidt orthogonalization leaves a
    # tiny negative squared residual norm due to float32 roundoff.
    # Two identical unit vectors -- the second residual becomes ~0 (or slightly negative).
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vectors = np.array([v, v + 1e-8, np.array([0, 1, 0], dtype=np.float32)], dtype=np.float32)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    result = greedy_log_det_select(vectors, query, k=2)
    # Must not raise and must not contain NaN-derived indices
    assert all(isinstance(i, int) for i in result)


# ---------------------------------------------------------------------------
# Input is not mutated
# ---------------------------------------------------------------------------

def test_input_vectors_not_mutated():
    """The function must not modify the input arrays."""
    rng = _make_rng(5)
    vectors = rng.standard_normal((8, 4)).astype(np.float32)
    query = rng.standard_normal(4).astype(np.float32)
    vectors_copy = vectors.copy()
    query_copy = query.copy()
    greedy_log_det_select(vectors, query, k=3)
    np.testing.assert_array_equal(vectors, vectors_copy)
    np.testing.assert_array_equal(query, query_copy)
