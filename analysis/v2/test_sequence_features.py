"""Small deterministic contract tests for v2 path extraction kernels."""

from __future__ import annotations

import numpy as np

from sequence_features import (
    N_BINS,
    N_MARKET_CHANNELS,
    N_ORDER_CHANNELS,
    N_TRADE_CHANNELS,
    _market_kernel,
    _order_kernel,
    _trade_kernel,
)


def run_synthetic_tests() -> None:
    offsets = np.array([0, 3, 4], dtype=np.int64)
    seconds = np.array([59.0, 5.0, 1.0, 2.0], dtype=np.float32)
    ref_mid = np.array([100.0, 50.0], dtype=np.float32)
    ref_spread = np.array([2.0, 1.0], dtype=np.float32)
    ref_depth = np.array([20.0, 10.0], dtype=np.float32)
    bid = np.array([98.0, 99.0, 99.0, 49.5], dtype=np.float32)
    ask = np.array([102.0, 101.0, 101.0, 50.5], dtype=np.float32)
    bid_volume = np.array([10, 12, 14, 7], dtype=np.int32)
    ask_volume = np.array([10, 8, 6, 3], dtype=np.int32)
    market = np.zeros((2, N_BINS * N_MARKET_CHANNELS), dtype=np.float32)
    _market_kernel(
        offsets,
        seconds,
        np.array([100.0, 100.0, 101.0, 50.0], dtype=np.float32),
        np.array([0, 10, 20, 5], dtype=np.int32),
        np.array([0, 1, 2, 1], dtype=np.int32),
        ask,
        ask_volume,
        bid,
        bid_volume,
        np.array([5, 5, 5, 2], dtype=np.int32),
        np.array([5, 5, 5, 2], dtype=np.int32),
        ref_mid,
        ref_spread,
        ref_depth,
        market,
    )
    assert np.isclose(market[0, 0], np.log1p(2.0))
    assert np.isclose(market[0, 9 * N_MARKET_CHANNELS], np.log1p(1.0))
    assert np.isclose(market[0, 3], 0.3)
    assert np.isnan(market[1, N_MARKET_CHANNELS + 1])

    flow = np.zeros(
        (2, N_BINS * (N_ORDER_CHANNELS + N_TRADE_CHANNELS)), dtype=np.float32
    )
    _order_kernel(
        offsets,
        seconds,
        np.array([98.0, 101.0, 99.0, 50.0], dtype=np.float32),
        np.array([10, 20, 30, 5], dtype=np.int32),
        np.array([0, 0, 1, 1], dtype=np.int8),
        np.array([0, 1, 0, 0], dtype=np.int8),
        ref_mid,
        ref_spread,
        market,
        flow,
    )
    base = 0
    assert np.isclose(flow[0, base], np.log1p(2.0))
    assert np.isclose(flow[0, base + 1], np.log1p(50.0))
    # buy-cancel and sell-new are both negative pressure.
    assert np.isclose(flow[0, base + 2], -1.0)
    assert np.isclose(flow[0, base + 6], 0.5)

    _trade_kernel(
        offsets,
        seconds,
        np.array([99.0, 101.0, 99.0, 50.0], dtype=np.float32),
        np.array([10, 20, 30, 5], dtype=np.int32),
        np.array([0, 0, 1, 1], dtype=np.int8),
        ref_mid,
        ref_spread,
        market,
        flow,
    )
    trade_base = N_BINS * N_ORDER_CHANNELS
    assert np.isclose(flow[0, trade_base], np.log1p(2.0))
    assert np.isclose(flow[0, trade_base + 2], 0.0)
    assert np.isclose(flow[0, trade_base + 3], -0.2)
    assert np.all(np.isfinite(flow[:, [0, trade_base]]))


if __name__ == "__main__":
    run_synthetic_tests()
    print("sequence feature synthetic tests passed")
