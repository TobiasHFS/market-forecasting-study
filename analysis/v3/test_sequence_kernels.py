"""Synthetic semantic tests; run directly with Python, no raw dataset required."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sequence_kernels import (asof_l1_quotes, depth_units, ofi_l1_events,
                              spread_units, valid_l1_quotes, weighted_bin_mean)


class SequenceKernelTests(unittest.TestCase):
    def test_valid_book_requires_prices_and_finite_positive_total_depth(self):
        np.testing.assert_array_equal(valid_l1_quotes(
            [99, 0, 101, 99, 99, 100, 99], [101, 0, 100, 101, 101, 100, 101],
            [10, 0, 10, 0, np.inf, 10, 0], [10, 0, 10, 0, 10, 10, 10]),
            [True, False, False, False, False, True, True])

    def test_zero_crossed_and_invalid_time_rows_do_not_seed_ofi(self):
        output = ofi_l1_events(np.array([0, 6]), np.array([9, 7, 5, 3, np.nan, 1]),
                              np.array([99, 0, 102, 99, 200, 99]),
                              np.array([101, 0, 101, 101, 202, 101]),
                              np.array([10, 0, 10, 10, 90, 10]),
                              np.array([10, 0, 10, 10, 90, 10]))
        np.testing.assert_allclose(output, [np.nan, np.nan, np.nan, 0, np.nan, 0], equal_nan=True)

    def test_ofi_price_moves_and_equal_price_depth_changes(self):
        output = ofi_l1_events(np.array([0, 4]), np.array([4, 3, 2, 1]),
                              np.array([99, 99, 100, 99]), np.array([101, 101, 102, 101]),
                              np.array([10, 14, 17, 11]), np.array([10, 8, 9, 12]))
        np.testing.assert_allclose(output, [np.nan, 6, 25, -29], equal_nan=True)

    def test_ofi_state_resets_between_samples(self):
        output = ofi_l1_events(np.array([0, 1, 1, 2]), np.array([1, 1]),
                              np.array([99, 199]), np.array([101, 201]),
                              np.array([10, 20]), np.array([10, 20]))
        self.assertTrue(np.isnan(output).all())

    def test_missing_reference_is_nan_and_measured_zero_is_zero(self):
        np.testing.assert_allclose(spread_units(
            [100] * 6, [100, 100, np.nan, 100, 100, 100], [2, 0, 2, -1, 1e-10, np.inf]),
            [0, np.nan, np.nan, np.nan, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(depth_units([0, 0, 1], [20, 0, np.nan]),
                                   [0, np.nan, np.nan], equal_nan=True)

    @staticmethod
    def join(queries, *, quotes=(11, 8, 2), mid=(100, 110, 999), **kwargs):
        mid = np.asarray(mid, dtype=float)
        return asof_l1_quotes(np.array([0, len(queries)]), np.asarray(queries),
                             np.array([0, len(quotes)]), np.asarray(quotes), mid - 1,
                             mid + 1, np.ones(mid.size), np.ones(mid.size), **kwargs)

    def test_backward_age_direction_no_future_quote_in_same_bin(self):
        joined = self.join([9, 5, 1, 12])
        np.testing.assert_allclose(joined.mid, [100, 110, 999, np.nan], equal_nan=True)
        np.testing.assert_array_equal(joined.row_index, [0, 1, 2, -1])
        np.testing.assert_allclose(joined.lag_seconds, [2, 3, 1, np.nan], equal_nan=True)

    def test_no_prior_does_not_fallback_to_terminal_mid(self):
        joined = self.join([5], quotes=(2,), mid=(999,))
        self.assertEqual(joined.row_index[0], -1)
        self.assertTrue(np.isnan(joined.mid[0]))

    def test_perturbing_or_appending_future_quotes_cannot_change_event_alignment(self):
        before = self.join([5], quotes=(11, 8), mid=(100, 110))
        after = self.join([5], quotes=(11, 8, 2, 0), mid=(100, 110, 10000, 50000))
        np.testing.assert_array_equal(before.mid, after.mid)
        np.testing.assert_array_equal(before.row_index, after.row_index)

    def test_ties_excluded_by_default_explicit_opt_in_selects_last_tied_quote(self):
        default = self.join([8], quotes=(11, 8, 8, 2), mid=(100, 110, 111, 999))
        exact = self.join([8], quotes=(11, 8, 8, 2), mid=(100, 110, 111, 999),
                          allow_exact_matches=True)
        np.testing.assert_array_equal(default.mid, [100])
        np.testing.assert_array_equal(exact.mid, [111])

    def test_stale_missing_and_post_prediction_queries(self):
        joined = self.join([5, 9, np.nan, -1], max_lag_seconds=2)
        np.testing.assert_allclose(joined.mid, [np.nan, 100, np.nan, np.nan], equal_nan=True)

    def test_alignment_cannot_cross_sample_boundary_or_use_invalid_quote(self):
        joined = asof_l1_quotes(np.array([0, 1, 2, 3]), np.array([1, 1, 1]),
                               np.array([0, 3, 3, 4]), np.array([9, 5, 2, 3]),
                               np.array([99, 0, 101, 199]), np.array([101, 0, 100, 201]),
                               np.array([10, 0, 10, 10]), np.array([10, 0, 10, 10]))
        np.testing.assert_allclose(joined.mid, [100, np.nan, 200], equal_nan=True)

    def test_locked_quote_can_align_but_cannot_supply_spread_units(self):
        joined = asof_l1_quotes(np.array([0, 1]), np.array([1]), np.array([0, 1]),
                               np.array([2]), np.array([100]), np.array([100]),
                               np.array([10]), np.array([10]))
        self.assertEqual(joined.mid[0], 100)
        self.assertTrue(np.isnan(spread_units([100], joined.mid, joined.spread)[0]))

    def test_invalid_time_and_infinite_volume_quotes_are_skipped(self):
        joined = asof_l1_quotes(np.array([0, 1]), np.array([1]), np.array([0, 4]),
                               np.array([9, np.nan, 3, -1]), np.array([99, 999, 999, 999]),
                               np.array([101, 1001, 1001, 1001]), np.array([10, 10, np.inf, 10]),
                               np.array([10, 10, 10, 10]))
        self.assertEqual(joined.mid[0], 100)

    def test_time_inversions_and_bad_offsets_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "nonincreasing"):
            self.join([1], quotes=(2, 8), mid=(100, 110))
        with self.assertRaisesRegex(ValueError, "offsets"):
            weighted_bin_mean(np.array([0, 3]), np.array([1]), np.array([1]), np.array([1]))
        with self.assertRaisesRegex(ValueError, "same samples"):
            asof_l1_quotes(np.array([0, 0, 1]), np.array([1]), np.array([0, 1]),
                           np.array([2]), np.array([99]), np.array([101]),
                           np.array([10]), np.array([10]))

    def test_weighted_mean_excludes_undefined_numerator_from_denominator(self):
        mean, weight = weighted_bin_mean(np.array([0, 3]), np.array([1, 2, 3]),
                                         np.array([2, np.nan, 4]), np.array([10, 1000, 30]))
        self.assertEqual(mean[0, 0], 3.5)
        self.assertEqual(weight[0, 0], 40)
        self.assertTrue(np.isnan(mean[0, 1:]).all())
        self.assertTrue((weight[0, 1:] == 0).all())

    def test_bin_boundaries_horizon_and_future_exclusion(self):
        mean, weight = weighted_bin_mean(np.array([0, 6]), np.array([0, 6, 60, 61, -1, np.nan]),
                                         np.array([1, 2, 3, 900, 800, 700]), np.ones(6))
        np.testing.assert_array_equal(mean[0, [0, 1, 9]], [1, 2, 3])
        self.assertEqual(weight.sum(), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
