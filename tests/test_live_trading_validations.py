import logging
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from live_trading import (
    StateStore,
    STATE_VERSION,
    filter_orderbook_entries,
    normalize_ws_trade_entry,
    require_valid_tp_sl,
    tp_sl_backoff_delay,
)


def test_require_valid_tp_sl_buy() -> None:
    require_valid_tp_sl("Buy", 100.0, 110.0, 90.0)


def test_require_valid_tp_sl_sell() -> None:
    require_valid_tp_sl("Sell", 100.0, 90.0, 110.0)


def test_require_valid_tp_sl_invalid() -> None:
    with pytest.raises(RuntimeError):
        require_valid_tp_sl("Buy", 100.0, 90.0, 110.0)


def test_tp_sl_backoff_delay() -> None:
    assert tp_sl_backoff_delay(1, 2.0, 2.0, 60.0) == 2.0
    assert tp_sl_backoff_delay(2, 2.0, 2.0, 60.0) == 4.0
    assert tp_sl_backoff_delay(5, 2.0, 2.0, 60.0) == 32.0
    assert tp_sl_backoff_delay(6, 2.0, 2.0, 60.0) == 60.0


@given(
    ts=st.integers(min_value=1, max_value=2_000_000_000_000),
    price=st.floats(min_value=0.0001, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    size=st.floats(min_value=0.0001, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
)
def test_normalize_ws_trade_entry_valid(ts: int, price: float, size: float) -> None:
    trade = {"T": ts, "p": price, "v": size, "S": "Buy", "i": "1"}
    result = normalize_ws_trade_entry(trade)
    assert result is not None
    assert result["price"] == pytest.approx(price)
    assert result["size"] == pytest.approx(size)
    assert result["side"] == "Buy"


def test_normalize_ws_trade_entry_invalid() -> None:
    assert normalize_ws_trade_entry({"T": 1, "p": -1.0, "v": 1.0}) is None


@given(
    entries=st.lists(
        st.tuples(
            st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
        ),
        max_size=50,
    )
)
def test_filter_orderbook_entries_sanitizes(entries) -> None:
    cleaned, _ = filter_orderbook_entries(entries, allow_zero_size=False)
    for price, size in cleaned:
        assert price > 0
        assert size > 0


def test_filter_orderbook_entries_zero_size() -> None:
    entries = [[1.0, 0.0], [2.0, 1.5]]
    cleaned, _ = filter_orderbook_entries(entries, allow_zero_size=True)
    assert (1.0, 0.0) in cleaned
    cleaned_strict, _ = filter_orderbook_entries(entries, allow_zero_size=False)
    assert (1.0, 0.0) not in cleaned_strict


def test_coerce_state_version_mismatch(tmp_path: Path) -> None:
    logger = logging.getLogger("test_state")
    store = StateStore(tmp_path / "state.json", "SYM", logger)
    default_state = store._default_state()
    state, errors = StateStore._coerce_state_payload(
        {"version": "bad", "symbol": "SYM"}, default_state, "SYM"
    )
    assert "version_mismatch" in errors
    assert state == default_state


def test_coerce_state_symbol_mismatch(tmp_path: Path) -> None:
    logger = logging.getLogger("test_state")
    store = StateStore(tmp_path / "state.json", "SYM", logger)
    default_state = store._default_state()
    state, errors = StateStore._coerce_state_payload(
        {"version": STATE_VERSION, "symbol": "OTHER"}, default_state, "SYM"
    )
    assert "symbol_mismatch" in errors
    assert state == default_state
