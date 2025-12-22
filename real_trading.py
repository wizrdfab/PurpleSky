"""
Real Trading Bot for TrendFollower

This module integrates the ML models with the Bybit exchange for live trading.
It replaces the paper trading simulation with real order execution.

⚠️ WARNING: This trades REAL MONEY. Use with extreme caution!
⚠️ ALWAYS test thoroughly on testnet before using real funds.

Usage:
    # Test on testnet first!
    python real_trading.py --symbol BTCUSDT --testnet
    
    # Live trading (REAL MONEY!)
    python real_trading.py --symbol BTCUSDT --live --model-dir ./models
"""
import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

# Local imports
from exchange_client import BybitClient, OrderResult, Position
from models import TrendFollowerModels
from config import TrendFollowerConfig, DEFAULT_CONFIG
from feature_engine import calculate_features_for_timeframe, get_feature_columns
from models import TrendFollowerModels

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('real_trading.log')
    ]
)
logger = logging.getLogger('RealTrading')


@dataclass
class TradeRecord:
    """Record of an executed trade"""
    timestamp: str
    symbol: str
    side: str
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    quality: str
    trend_prob: float
    bounce_prob: float
    order_id: str
    status: str  # "open", "closed_tp", "closed_sl", "closed_manual"
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None


class RealTradingBot:
    """
    Real trading bot that executes trades based on ML model signals.
    
    This bot:
    1. Fetches real-time market data from Bybit
    2. Calculates features from the data
    3. Gets predictions from ML models
    4. Executes trades when conditions are met
    5. Manages positions with stop loss and take profit
    
    ⚠️ RISK WARNING:
    - Always start with testnet
    - Use small position sizes
    - Monitor the bot closely
    - Have emergency stop procedures ready
    """
    
    # Default trading parameters
    DEFAULT_PARAMS = {
        'position_size_pct': 0.02,  # 2% of capital per trade
        'stop_loss_atr': 1.0,       # Stop loss in ATR multiples
        'take_profit_rr': 2.0,      # Take profit ratio (2:1 R:R)
        'min_quality': 'B',         # Minimum signal quality
        'min_trend_prob': 0.5,      # Minimum trend probability
        'min_bounce_prob': 0.5,     # Minimum bounce probability
        'max_positions': 1,         # Maximum concurrent positions
        'cooldown_bars': 2,         # Bars to wait after closing
        'leverage': 1,              # Default leverage
    }
    
    def __init__(
        self,
        exchange_client: BybitClient,
        models: TrendFollowerModels,
        config: TrendFollowerConfig = DEFAULT_CONFIG,
        params: Dict = None,
        results_dir: str = './real_results',
    ):
        self.client = exchange_client
        self.models = models
        self.config = config
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # State
        self.current_position: Optional[Dict] = None
        self.trade_history: List[TradeRecord] = []
        self.cooldown_until = 0
        self.bars_processed = 0
        
        # Session tracking
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_start = datetime.now()
        
        # Base timeframe
        self.base_tf = config.features.timeframe_names[config.base_timeframe_idx]
        
        # Feature columns (will be set after first feature calculation)
        self.feature_cols = None
        
        logger.info(f"RealTradingBot initialized")
        logger.info(f"  Testnet: {exchange_client.testnet}")
        logger.info(f"  Base TF: {self.base_tf}")
        logger.info(f"  Min Quality: {self.params['min_quality']}")
        logger.info(f"  Position Size: {self.params['position_size_pct']:.1%}")
        logger.info(f"  R:R Ratio: 1:{self.params['take_profit_rr']}")
    
    def check_existing_position(self, symbol: str) -> bool:
        """Check if we have an existing position."""
        position = self.client.get_position(symbol)
        
        if position and position.is_open:
            self.current_position = {
                'symbol': symbol,
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
            }
            logger.info(f"Existing position found: {position.side} {position.size} @ {position.entry_price}")
            return True
        else:
            self.current_position = None
            return False
    
    def fetch_market_data(self, symbol: str, interval: str = "5", limit: int = 200) -> pd.DataFrame:
        """
        Fetch recent kline/candlestick data from Bybit.
        
        Args:
            symbol: Trading pair
            interval: Timeframe ("1", "5", "15", "60", "240", "D")
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            response = self.client.session.get_kline(
                category=self.client.category,
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if response['retCode'] != 0:
                logger.error(f"Failed to fetch klines: {response['retMsg']}")
                return pd.DataFrame()
            
            data = response['result']['list']
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'bar_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert types
            df['bar_time'] = pd.to_numeric(df['bar_time']) / 1000  # ms to seconds
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col])
            
            # Sort by time ascending
            df = df.sort_values('bar_time').reset_index(drop=True)
            
            # Add datetime column
            df['datetime'] = pd.to_datetime(df['bar_time'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from OHLCV data."""
        if len(df) < 100:
            logger.warning(f"Not enough data for feature calculation: {len(df)} bars")
            return pd.DataFrame()
        
        try:
            # Calculate features
            featured = calculate_features_for_timeframe(df, self.config.features)
            
            # Add timeframe prefix
            feature_cols = [c for c in featured.columns 
                          if c not in ['bar_time', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            rename_map = {c: f'{self.base_tf}_{c}' for c in feature_cols}
            featured = featured.rename(columns=rename_map)
            
            # Get feature columns if not set
            if self.feature_cols is None:
                self.feature_cols = get_feature_columns(featured)
            
            return featured
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return pd.DataFrame()
    
    def get_signal(self, features: pd.DataFrame) -> Dict:
        """
        Get trading signal from ML models.
        
        Returns:
            Dict with signal information:
            - direction: 1 (long), -1 (short), 0 (no trade)
            - quality: 'A', 'B', 'C'
            - trend_prob: probability of trend
            - bounce_prob: probability of bounce
            - should_trade: whether to execute trade
        """
        if features.empty or self.feature_cols is None:
            return {'direction': 0, 'should_trade': False}
        
        # Get latest bar
        latest = features.iloc[[-1]]
        
        # Prepare feature matrix
        X = latest[self.feature_cols].fillna(0)
        
        # Get predictions
        trend_pred = self.models.trend_classifier.predict(X)
        entry_pred = self.models.entry_model.predict(X)
        
        trend_dir = trend_pred['prediction'][0]
        prob_up = trend_pred['prob_up'][0]
        prob_down = trend_pred['prob_down'][0]
        bounce_prob = entry_pred['bounce_prob'][0]
        
        # Get alignment
        alignment_col = f'{self.base_tf}_ema_alignment'
        alignment = latest[alignment_col].iloc[0] if alignment_col in latest.columns else 0
        
        # Check if in pullback zone
        atr_col = f'{self.base_tf}_atr'
        ema_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}'
        
        if atr_col in latest.columns and ema_col in latest.columns:
            price = latest['close'].iloc[0]
            atr = latest[atr_col].iloc[0]
            ema = latest[ema_col].iloc[0]
            dist_from_ema = abs(price - ema) / atr if atr > 0 else 999
            is_pullback = dist_from_ema <= self.config.labels.pullback_threshold
        else:
            is_pullback = False
        
        # Determine trend alignment
        trend_aligned = (trend_dir == np.sign(alignment)) and trend_dir != 0
        
        # Determine quality
        if bounce_prob > 0.6 and trend_aligned and is_pullback:
            quality = 'A'
        elif bounce_prob > 0.5 and (trend_aligned or is_pullback):
            quality = 'B'
        else:
            quality = 'C'
        
        # Get trend probability
        if trend_dir == 1:
            trend_prob = prob_up
        elif trend_dir == -1:
            trend_prob = prob_down
        else:
            trend_prob = 0.0
        
        # Check if should trade
        quality_ok = ord(quality) <= ord(self.params['min_quality'])
        trend_prob_ok = trend_prob >= self.params['min_trend_prob']
        bounce_ok = bounce_prob >= self.params['min_bounce_prob']
        
        should_trade = (
            trend_dir != 0 and
            quality_ok and
            trend_prob_ok and
            bounce_ok
        )
        
        return {
            'direction': int(trend_dir),
            'quality': quality,
            'trend_prob': float(trend_prob),
            'bounce_prob': float(bounce_prob),
            'is_pullback': is_pullback,
            'trend_aligned': trend_aligned,
            'should_trade': should_trade,
        }

    def calculate_position_size(self, symbol: str, stop_distance: float) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Trading pair
            stop_distance: Distance to stop loss in price units
            
        Returns:
            Position size in base currency
        """
        # Get available balance
        balance = self.client.get_available_balance()
        
        if balance <= 0:
            logger.error("No available balance!")
            return 0.0
        
        # Calculate risk amount
        risk_amount = balance * self.params['position_size_pct']
        
        # Calculate position size
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0
        
        # Get instrument info for min qty and qty step
        info = self.client.get_instrument_info(symbol)
        if info:
            min_qty = float(info.get('lotSizeFilter', {}).get('minOrderQty', 0))
            qty_step = float(info.get('lotSizeFilter', {}).get('qtyStep', 0.001))
            
            # Round to qty step
            if qty_step > 0:
                position_size = round(position_size / qty_step) * qty_step
            
            # Ensure minimum
            position_size = max(position_size, min_qty)
        
        return position_size
    
    def execute_trade(self, symbol: str, signal: Dict, current_price: float, atr: float) -> bool:
        """
        Execute a trade based on signal.
        
        Args:
            symbol: Trading pair
            signal: Signal dict from get_signal()
            current_price: Current market price
            atr: Current ATR value
            
        Returns:
            True if trade was executed successfully
        """
        direction = signal['direction']
        side = "Buy" if direction == 1 else "Sell"
        
        # Calculate stop loss and take profit
        stop_distance = self.params['stop_loss_atr'] * atr
        
        if direction == 1:  # Long
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * self.params['take_profit_rr'])
        else:  # Short
            stop_loss = current_price + stop_distance
            take_profit = current_price - (stop_distance * self.params['take_profit_rr'])
        
        # Calculate position size
        qty = self.calculate_position_size(symbol, stop_distance)
        
        if qty <= 0:
            logger.error("Position size is 0, cannot trade")
            return False
        
        # Set leverage
        self.client.set_leverage(symbol, self.params['leverage'])
        
        # Place order with SL/TP
        logger.info(f"Executing trade: {side} {qty} {symbol}")
        logger.info(f"  Entry: {current_price:.6f}")
        logger.info(f"  Stop Loss: {stop_loss:.6f}")
        logger.info(f"  Take Profit: {take_profit:.6f}")
        logger.info(f"  Quality: {signal['quality']}, Trend: {signal['trend_prob']:.1%}, Bounce: {signal['bounce_prob']:.1%}")
        
        result = self.client.open_position(
            symbol=symbol,
            side=side,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self.params['leverage'],
        )
        
        if result.success:
            # Record the trade
            trade = TradeRecord(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quality=signal['quality'],
                trend_prob=signal['trend_prob'],
                bounce_prob=signal['bounce_prob'],
                order_id=result.order_id,
                status="open",
            )
            self.trade_history.append(trade)
            
            # Update current position
            self.current_position = {
                'symbol': symbol,
                'side': side,
                'size': qty,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'order_id': result.order_id,
                'trade_record': trade,
            }
            
            logger.info(f"✅ Trade executed successfully! Order ID: {result.order_id}")
            return True
        else:
            logger.error(f"❌ Trade failed: {result.error_message}")
            return False
    
    def check_position_status(self, symbol: str) -> str:
        """
        Check if position is still open or has been closed by SL/TP.
        
        Returns:
            "open", "closed_tp", "closed_sl", "closed_manual", or "no_position"
        """
        position = self.client.get_position(symbol)
        
        if not position or not position.is_open:
            if self.current_position:
                # Position was closed - determine how
                # (In real implementation, you'd check order history)
                return "closed_tp"  # Simplified
            return "no_position"
        
        return "open"
    
    def update_trade_record(self, symbol: str, status: str, exit_price: float = None):
        """Update the trade record when position is closed."""
        if not self.trade_history:
            return
        
        # Find the most recent trade for this symbol
        for trade in reversed(self.trade_history):
            if trade.symbol == symbol and trade.status == "open":
                trade.status = status
                trade.exit_time = datetime.now().isoformat()
                trade.exit_price = exit_price or self.client.get_current_price(symbol)
                
                # Calculate P&L
                if trade.side == "Buy":
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.qty
                else:
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.qty
                
                logger.info(f"Trade closed: {status}, P&L: ${trade.pnl:.2f}")
                break
    
    def save_results(self):
        """Save trade history and session stats."""
        # Save trades
        trades_file = self.results_dir / f'trades_{self.session_id}.json'
        trades_data = [asdict(t) for t in self.trade_history]
        with open(trades_file, 'w') as f:
            json.dump(trades_data, f, indent=2)
        
        # Calculate stats
        closed_trades = [t for t in self.trade_history if t.status != "open"]
        wins = [t for t in closed_trades if t.pnl and t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl and t.pnl <= 0]
        
        stats = {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'bars_processed': self.bars_processed,
            'total_trades': len(self.trade_history),
            'closed_trades': len(closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(closed_trades) if closed_trades else 0,
            'total_pnl': sum(t.pnl for t in closed_trades if t.pnl),
            'avg_win': sum(t.pnl for t in wins if t.pnl) / len(wins) if wins else 0,
            'avg_loss': sum(t.pnl for t in losses if t.pnl) / len(losses) if losses else 0,
            'params': self.params,
        }
        
        stats_file = self.results_dir / f'stats_{self.session_id}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def run_iteration(self, symbol: str) -> bool:
        """
        Run one iteration of the trading loop.
        
        Returns:
            True if should continue, False if should stop
        """
        self.bars_processed += 1
        
        try:
            # Check existing position
            position_exists = self.check_existing_position(symbol)
            
            if position_exists:
                # Check if position was closed by SL/TP
                status = self.check_position_status(symbol)
                
                if status != "open":
                    self.update_trade_record(symbol, status)
                    self.current_position = None
                    self.cooldown_until = self.bars_processed + self.params['cooldown_bars']
                    logger.info(f"Position closed, cooldown for {self.params['cooldown_bars']} bars")
            
            # Skip if in cooldown
            if self.bars_processed < self.cooldown_until:
                logger.debug(f"In cooldown, skipping signal check")
                return True
            
            # Skip if already have max positions
            if position_exists:
                return True
            
            # Fetch market data
            df = self.fetch_market_data(symbol, interval="5", limit=200)
            
            if df.empty:
                logger.warning("Failed to fetch market data")
                return True
            
            # Calculate features
            features = self.calculate_features(df)
            
            if features.empty:
                logger.warning("Failed to calculate features")
                return True
            
            # Get signal
            signal = self.get_signal(features)
            
            logger.debug(
                f"Signal: dir={signal['direction']}, quality={signal.get('quality', 'N/A')}, "
                f"trend={signal.get('trend_prob', 0):.1%}, bounce={signal.get('bounce_prob', 0):.1%}"
            )
            
            # Execute trade if signal is good
            if signal['should_trade']:
                current_price = df['close'].iloc[-1]
                atr_col = f'{self.base_tf}_atr'
                atr = features[atr_col].iloc[-1] if atr_col in features.columns else current_price * 0.02
                
                self.execute_trade(symbol, signal, current_price, atr)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Error in iteration: {e}")
            return True
    
    def run(self, symbol: str, interval_seconds: int = 60):
        """
        Main trading loop.
        
        Args:
            symbol: Trading pair to trade
            interval_seconds: Seconds between iterations
        """
        logger.info(f"Starting real trading bot for {symbol}")
        logger.info(f"Interval: {interval_seconds}s")
        logger.info("Press Ctrl+C to stop")
        
        if not self.client.testnet:
            logger.warning("⚠️ LIVE TRADING MODE - REAL MONEY AT RISK!")
            response = input("Type 'YES' to confirm: ")
            if response != "YES":
                logger.info("Aborted by user")
                return
        
        try:
            while True:
                start_time = time.time()
                
                should_continue = self.run_iteration(symbol)
                
                if not should_continue:
                    break
                
                # Wait for next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Stopping bot...")
        finally:
            self.save_results()
            logger.info("Bot stopped")


def main():
    parser = argparse.ArgumentParser(description='Real Trading Bot for TrendFollower')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Directory with trained models')
    parser.add_argument('--testnet', action='store_true', default=True,
                       help='Use testnet (default: True)')
    parser.add_argument('--live', action='store_true',
                       help='Use live trading (REAL MONEY!)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Seconds between iterations (default: 60)')
    parser.add_argument('--min-quality', type=str, default='B',
                       choices=['A', 'B', 'C'],
                       help='Minimum signal quality (default: B)')
    parser.add_argument('--position-size', type=float, default=0.02,
                       help='Position size as fraction of capital (default: 0.02)')
    parser.add_argument('--leverage', type=int, default=1,
                       help='Trading leverage (default: 1)')
    
    args = parser.parse_args()
    
    # Determine testnet mode
    use_testnet = not args.live
    
    print("=" * 60)
    print("TRENDFOLOWER REAL TRADING BOT")
    print("=" * 60)
    print(f"Symbol:        {args.symbol}")
    print(f"Mode:          {'TESTNET' if use_testnet else '⚠️ LIVE TRADING ⚠️'}")
    print(f"Model Dir:     {args.model_dir}")
    print(f"Min Quality:   {args.min_quality}")
    print(f"Position Size: {args.position_size:.1%}")
    print(f"Leverage:      {args.leverage}x")
    print("=" * 60)
    
    # Check for API keys
    api_key = os.environ.get('BYBIT_API_KEY')
    api_secret = os.environ.get('BYBIT_API_SECRET')
    
    if not api_key or not api_secret:
        print("\n❌ Error: API keys not found!")
        print("Set environment variables:")
        print("  export BYBIT_API_KEY='your_api_key'")
        print("  export BYBIT_API_SECRET='your_api_secret'")
        sys.exit(1)
    
    # Load models
    print("\nLoading models...")
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"❌ Error: Model directory not found: {model_path}")
        sys.exit(1)
    
    config = DEFAULT_CONFIG
    config.model.model_dir = model_path
    
    models = TrendFollowerModels(config.model)
    try:
        models.load_all(model_path)
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)
    
    # Create exchange client
    print("\nConnecting to Bybit...")
    client = BybitClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=use_testnet,
        category="linear"
    )
    
    # Verify connection
    balance = client.get_available_balance()
    print(f"✓ Connected! Available balance: ${balance:,.2f} USDT")
    
    # Create trading bot
    params = {
        'min_quality': args.min_quality,
        'position_size_pct': args.position_size,
        'leverage': args.leverage,
    }
    
    bot = RealTradingBot(
        exchange_client=client,
        models=models,
        config=config,
        params=params,
    )
    
    # Start trading
    print("\nStarting trading bot...")
    bot.run(args.symbol, interval_seconds=args.interval)


if __name__ == "__main__":
    main()
