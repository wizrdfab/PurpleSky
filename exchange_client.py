"""
Lightweight Bybit HTTP client wrapper for TrendFollower live trading.

Provides minimal calls needed by the live trader:
- wallet balance (robust to string/float conversions)
- instrument info (lot size / qty step)
- market price
- open position with stop loss / take profit
- fetch current position

Uses pybit unified_trading REST client under the hood.
"""
from dataclasses import dataclass
from typing import Optional
from decimal import Decimal, InvalidOperation

try:
    from pybit.unified_trading import HTTP
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("pybit is required: pip install pybit") from exc


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class Position:
    is_open: bool
    side: str = ""
    size: float = 0.0
    entry_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class BybitClient:
    """Small convenience wrapper around pybit.HTTP for unified trading."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, category: str = "linear"):
        self.testnet = testnet
        self.category = category
        self.session = HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)

    # ------------------------------------------------------------------ utils
    def _to_float(self, value, default: float = 0.0) -> float:
        """Convert Bybit numeric responses (often strings) to float safely."""
        try:
            return float(value)
        except (TypeError, ValueError, InvalidOperation):
            try:
                return float(Decimal(str(value)))
            except Exception:
                return default

    # ---------------------------------------------------------------- balance
    def _extract_balance(self, resp, asset: str) -> Optional[float]:
        if not resp or resp.get("retCode", -1) != 0:
            return None
        balances = resp.get("result", {}).get("list", [])
        if not balances:
            return None
        coin_info = balances[0].get("coin", [])
        for item in coin_info:
            if item.get("coin") == asset:
                available = (
                    item.get("availableToWithdraw")
                    or item.get("availableBalance")
                    or item.get("walletBalance")
                    or 0
                )
                return self._to_float(available, None)
        return None

    def get_available_balance(self, asset: str = "USDT", account_types: Optional[list] = None, logger=None) -> float:
        """
        Return available balance for an asset.
        Tries multiple account types and logs retCode/retMsg on failure when logger is provided.
        """
        account_types = account_types or ["UNIFIED", "CONTRACT", "SPOT"]
        for acct in account_types:
            try:
                resp = self.session.get_wallet_balance(accountType=acct, coin=asset)
                bal = self._extract_balance(resp, asset)
                if bal is not None:
                    return bal
                if logger:
                    logger.warning(f"Balance attempt {acct} retCode={resp.get('retCode')} retMsg={resp.get('retMsg')}")
            except Exception as exc:
                if logger:
                    logger.warning(f"Balance attempt {acct} error: {exc}")
                continue
        return 0.0

    # --------------------------------------------------------------- instrument
    def get_instrument_info(self, symbol: str) -> Optional[dict]:
        """Fetch instrument filters (qty steps, min qty, etc.)."""
        try:
            resp = self.session.get_instruments_info(category=self.category, symbol=symbol)
            if resp.get("retCode", -1) != 0:
                return None
            data = resp.get("result", {}).get("list", [])
            return data[0] if data else None
        except Exception:
            return None

    # ------------------------------------------------------------------ prices
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest trade/mark price."""
        try:
            resp = self.session.get_tickers(category=self.category, symbol=symbol)
            if resp.get("retCode", -1) != 0:
                return None
            data = resp.get("result", {}).get("list", [])
            if not data:
                return None
            price = data[0].get("lastPrice") or data[0].get("markPrice")
            return self._to_float(price, None)
        except Exception:
            return None

    # ---------------------------------------------------------------- positions
    def get_position(self, symbol: str) -> Optional[Position]:
        """Return current position if open, else None."""
        try:
            resp = self.session.get_positions(category=self.category, symbol=symbol)
            if resp.get("retCode", -1) != 0:
                return None

            positions = resp.get("result", {}).get("list", [])
            for pos in positions:
                size = self._to_float(pos.get("size"), 0.0)
                if size == 0:
                    continue
                side = pos.get("side", "")
                entry = self._to_float(pos.get("avgPrice"), 0.0)
                stop_loss = pos.get("stopLoss")
                take_profit = pos.get("takeProfit")
                return Position(
                    is_open=True,
                    side=side,
                    size=size,
                    entry_price=entry,
                    stop_loss=self._to_float(stop_loss) if stop_loss else None,
                    take_profit=self._to_float(take_profit) if take_profit else None,
                )
        except Exception:
            return None
        return None

    # ---------------------------------------------------------------- leverage
    def set_leverage(self, symbol: str, leverage: int = 1) -> bool:
        """Set isolated leverage for a symbol."""
        try:
            resp = self.session.set_leverage(category=self.category, symbol=symbol, buyLeverage=leverage, sellLeverage=leverage)
            return resp.get("retCode", -1) == 0
        except Exception:
            return False

    # ------------------------------------------------------------------ orders
    def open_position(self, symbol: str, side: str, qty: float, stop_loss: float, take_profit: float, leverage: int = 1) -> OrderResult:
        """
        Open a market position with attached stop loss and take profit.

        Args:
            symbol: trading pair
            side: "Buy" or "Sell"
            qty: order size
            stop_loss: absolute price
            take_profit: absolute price
            leverage: leverage to apply
        """
        try:
            self.set_leverage(symbol, leverage)
            resp = self.session.place_order(
                category=self.category,
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=qty,
                timeInForce="IOC",
                takeProfit=str(take_profit),
                stopLoss=str(stop_loss),
                reduceOnly=False,
            )
            if resp.get("retCode", -1) != 0:
                return OrderResult(success=False, error_message=resp.get("retMsg"))
            order_id = resp.get("result", {}).get("orderId")
            return OrderResult(success=True, order_id=order_id)
        except Exception as exc:
            return OrderResult(success=False, error_message=str(exc))

    # ----------------------------------------------------------- balance debug
    def get_balances_raw(self, asset: str = "USDT") -> list:
        """
        Return raw balance responses for troubleshooting.
        """
        results = []
        for acct in ["UNIFIED", "CONTRACT", "SPOT"]:
            try:
                resp = self.session.get_wallet_balance(accountType=acct, coin=asset)
            except Exception as exc:
                results.append({"accountType": acct, "error": str(exc)})
                continue
            coins = []
            try:
                coins = resp.get("result", {}).get("list", [])[0].get("coin", [])
            except Exception:
                pass
            results.append(
                {
                    "accountType": acct,
                    "retCode": resp.get("retCode"),
                    "retMsg": resp.get("retMsg"),
                    "coin_count": len(coins),
                    "coins": coins,
                }
            )
        return results
