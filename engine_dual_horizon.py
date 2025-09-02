"""
Dual-Horizon Enhanced Trading Environment for Crypto Trading RL

Revolutionary architecture combining:
1. Forward-looking "Promise" rewards based on future price movements
2. Backward-looking immediate realized PnL tracking
3. Precise 20-second granular decision making
4. 3-minute reward evaluation horizon

This creates an agent that learns both short-term execution and long-term positioning.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
import logging
from tqdm import tqdm
import warnings
from scipy import stats
from sklearn.preprocessing import RobustScaler

# Import the dual-horizon configuration
from config_dual_horizon import SETTINGS, FeatureKeys, DualHorizonConfig
from normalizer import Normalizer
from features import (
    StatefulBBWPercentRank,
    StatefulPriceDistanceMA,
    StatefulSRDistances,
)

logger = logging.getLogger(__name__)

# --- DUAL-HORIZON SPECIFIC DATA STRUCTURES ---

class Promise(NamedTuple):
    """Represents a promise made to the agent about future rewards."""
    step: int
    action_signal: float
    action_size: float
    entry_price: float
    promise_price: float
    target_step: int
    expected_return: float
    position_value: float

class RealizedTrade(NamedTuple):
    """Represents a completed trade with immediate PnL."""
    step: int
    entry_price: float
    exit_price: float
    position_size: float
    realized_pnl: float
    hold_duration: int

class DualHorizonRewardCalculator:
    """
    ✅ NEW: Advanced reward calculator implementing the dual-horizon system.
    
    This combines:
    1. Immediate rewards for realized PnL (backward-looking)
    2. Delayed "promise" rewards based on future price movements (forward-looking)
    """
    
    def __init__(self, config: DualHorizonConfig, leverage: float = 10.0):
        self.config = config
        self.leverage = leverage
        
        # Promise tracking system
        self.promises: Dict[int, Promise] = {}  # step -> Promise
        self.fulfilled_promises: List[Tuple[Promise, float]] = []
        
        # Realized PnL tracking
        self.realized_trades: List[RealizedTrade] = []
        self.total_realized_pnl = 0.0
        
        # Performance metrics
        self.promise_success_rate = 0.0
        self.avg_promise_return = 0.0
        self.realized_pnl_volatility = 0.0
        
        # Scaling factors
        self.immediate_scaling = config.immediate_reward_scaling
        self.delayed_scaling = config.delayed_reward_scaling
        
        logger.info(f"Dual-horizon reward calculator initialized:")
        logger.info(f"  Reward horizon: {config.reward_horizon_steps} steps")
        logger.info(f"  Immediate weight: {config.immediate_reward_weight:.1%}")
        logger.info(f"  Delayed weight: {config.delayed_reward_weight:.1%}")
        logger.info(f"  Max promises: {config.max_promises}")
    
    def create_promise(self, step: int, action: np.ndarray, current_price: float, 
                      portfolio_value: float) -> Optional[Promise]:
        """Create a new promise for future evaluation."""
        try:
            # Clean up old promises first
            self._cleanup_old_promises(step)
            
            # Check if we can create a new promise
            if len(self.promises) >= self.config.max_promises:
                # Remove oldest promise to make room
                oldest_step = min(self.promises.keys())
                del self.promises[oldest_step]
            
            action_signal = float(action[0])
            action_size = float(action[1])
            
            # Only create promises for significant actions
            if abs(action_signal) < 0.1 or action_size < 0.1:
                return None
            
            target_step = step + self.config.reward_horizon_steps
            
            # Calculate expected return based on signal strength
            expected_return = action_signal * action_size * 0.02  # 2% base expectation
            
            promise = Promise(
                step=step,
                action_signal=action_signal,
                action_size=action_size,
                entry_price=current_price,
                promise_price=current_price,
                target_step=target_step,
                expected_return=expected_return,
                position_value=portfolio_value * action_size * abs(action_signal)
            )
            
            self.promises[step] = promise
            return promise
            
        except Exception as e:
            logger.error(f"Error creating promise: {e}")
            return None
    
    def evaluate_promises(self, current_step: int, current_price: float) -> float:
        """Evaluate all promises that have reached their target step."""
        total_promise_reward = 0.0
        fulfilled_count = 0
        
        try:
            promises_to_fulfill = []
            
            # Find promises ready for evaluation
            for step, promise in self.promises.items():
                if current_step >= promise.target_step:
                    promises_to_fulfill.append(promise)
            
            # Evaluate each promise
            for promise in promises_to_fulfill:
                # Calculate actual return
                price_change = (current_price - promise.entry_price) / promise.entry_price
                
                # Adjust for position direction
                actual_return = price_change * np.sign(promise.action_signal)
                
                # Calculate promise reward
                # Positive if the agent's prediction was correct, negative otherwise
                expectation_error = actual_return - promise.expected_return
                
                # Scale reward by position size and confidence
                base_reward = actual_return * promise.action_size
                
                # Bonus/penalty for accuracy
                accuracy_bonus = expectation_error * promise.action_size * 0.5
                
                promise_reward = (base_reward + accuracy_bonus) * self.delayed_scaling
                
                # Apply bounds
                promise_reward = np.clip(promise_reward, -2.0, 2.0)
                
                total_promise_reward += promise_reward
                fulfilled_count += 1
                
                # Store fulfilled promise for analytics
                self.fulfilled_promises.append((promise, promise_reward))
                
                # Remove fulfilled promise
                del self.promises[promise.step]
            
            # Update promise analytics
            if fulfilled_count > 0 and self.config.enable_promise_analytics:
                self._update_promise_analytics()
            
            return total_promise_reward
            
        except Exception as e:
            logger.error(f"Error evaluating promises: {e}")
            return 0.0
    
    def calculate_realized_pnl_reward(self, trade: Optional[RealizedTrade]) -> float:
        """Calculate immediate reward for realized PnL."""
        if trade is None:
            return 0.0
        
        try:
            # Basic PnL reward
            pnl_reward = trade.realized_pnl / 10000.0  # Scale down large PnL values
            
            # Duration bonus/penalty
            optimal_hold_duration = 5  # 5 steps = 100 seconds
            duration_factor = 1.0
            if trade.hold_duration < optimal_hold_duration:
                duration_factor = 0.8  # Penalty for very short holds
            elif trade.hold_duration > optimal_hold_duration * 3:
                duration_factor = 0.9  # Small penalty for very long holds
            
            # Scale by immediate scaling factor
            immediate_reward = pnl_reward * duration_factor * self.immediate_scaling
            
            # Apply bounds
            immediate_reward = np.clip(immediate_reward, -1.0, 1.0)
            
            # Update realized PnL tracking
            self.realized_trades.append(trade)
            self.total_realized_pnl += trade.realized_pnl
            
            return immediate_reward
            
        except Exception as e:
            logger.error(f"Error calculating realized PnL reward: {e}")
            return 0.0
    
    def calculate_total_reward(self, immediate_reward: float, delayed_reward: float) -> float:
        """Combine immediate and delayed rewards according to configuration weights."""
        total_reward = (
            immediate_reward * self.config.immediate_reward_weight +
            delayed_reward * self.config.delayed_reward_weight
        )
        
        # Final bounds check
        return np.clip(total_reward, -3.0, 3.0)
    
    def _cleanup_old_promises(self, current_step: int):
        """Remove promises that are too old to be useful."""
        max_age = self.config.reward_horizon_steps * 2
        cutoff_step = current_step - max_age
        
        old_promises = [step for step in self.promises.keys() if step < cutoff_step]
        for step in old_promises:
            del self.promises[step]
    
    def _update_promise_analytics(self):
        """Update promise performance analytics."""
        if not self.fulfilled_promises:
            return
        
        try:
            recent_promises = self.fulfilled_promises[-50:]  # Last 50 promises
            
            # Calculate success rate
            successful = sum(1 for _, reward in recent_promises if reward > self.config.promise_success_threshold)
            self.promise_success_rate = successful / len(recent_promises)
            
            # Calculate average return
            self.avg_promise_return = np.mean([reward for _, reward in recent_promises])
            
            # Calculate realized PnL volatility
            if len(self.realized_trades) > 10:
                recent_pnl = [trade.realized_pnl for trade in self.realized_trades[-20:]]
                self.realized_pnl_volatility = np.std(recent_pnl)
            
        except Exception as e:
            logger.error(f"Error updating promise analytics: {e}")
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about dual-horizon performance."""
        return {
            'active_promises': len(self.promises),
            'fulfilled_promises': len(self.fulfilled_promises),
            'promise_success_rate': self.promise_success_rate,
            'avg_promise_return': self.avg_promise_return,
            'total_realized_pnl': self.total_realized_pnl,
            'realized_pnl_volatility': self.realized_pnl_volatility,
            'total_realized_trades': len(self.realized_trades),
            'immediate_weight': self.config.immediate_reward_weight,
            'delayed_weight': self.config.delayed_reward_weight
        }

class EnhancedRiskManager:
    """Advanced risk management system with dual-horizon awareness."""
    
    def __init__(self, config, leverage: float = 10.0):
        self.cfg = config
        self.leverage = leverage
        self.max_heat = 0.25
        self.volatility_lookback = 50
        self.risk_free_rate = 0.02
        self.volatility_buffer = deque(maxlen=self.volatility_lookback)
        self.return_buffer = deque(maxlen=100)
        
        # Dual-horizon specific risk management
        self.promise_risk_factor = 1.0
        self.realized_pnl_risk_factor = 1.0

    def update_market_regime(self, returns: np.ndarray, volatility: float) -> str:
        """Detect current market regime."""
        if len(returns) < 20:
            return "UNCERTAIN"
        
        recent_returns = returns[-20:]
        vol_percentile = np.percentile(self.volatility_buffer, 80) if len(self.volatility_buffer) > 10 else volatility
        
        if volatility > vol_percentile * 1.5:
            return "HIGH_VOLATILITY"
        elif np.mean(recent_returns) > 0.001 and volatility < vol_percentile * 0.8:
            return "TRENDING_UP"
        elif np.mean(recent_returns) < -0.001 and volatility < vol_percentile * 0.8:
            return "TRENDING_DOWN"
        elif volatility < vol_percentile * 0.6:
            return "LOW_VOLATILITY"
        else:
            return "SIDEWAYS"

    def calculate_dynamic_position_limit(self, volatility: float, portfolio_value: float,
                                       market_regime: str, promise_analytics: Dict) -> float:
        """Calculate dynamic position limits with dual-horizon considerations."""
        base_limit = self.cfg.strategy.max_position_size
        
        # Volatility adjustment
        vol_adjustment = max(0.5, min(2.0, 1.0 / (volatility * 10 + 0.1)))
        
        # Market regime adjustment
        regime_multipliers = {
            "HIGH_VOLATILITY": 0.6,
            "TRENDING_UP": 1.2,
            "TRENDING_DOWN": 0.8,
            "LOW_VOLATILITY": 1.1,
            "SIDEWAYS": 0.9,
            "UNCERTAIN": 0.7
        }
        regime_adjustment = regime_multipliers.get(market_regime, 0.8)
        
        # Leverage adjustment
        leverage_adjustment = min(1.0, 10.0 / self.leverage)
        
        # ✅ NEW: Dual-horizon adjustment
        dual_horizon_adjustment = 1.0
        if promise_analytics.get('promise_success_rate', 0.5) > 0.6:
            dual_horizon_adjustment = 1.1  # Boost if promises are working well
        elif promise_analytics.get('promise_success_rate', 0.5) < 0.4:
            dual_horizon_adjustment = 0.9  # Reduce if promises are failing
        
        return base_limit * vol_adjustment * regime_adjustment * leverage_adjustment * dual_horizon_adjustment

# Map calculator names from config to their classes
STATEFUL_CALCULATOR_MAP = {
    'StatefulBBWPercentRank': StatefulBBWPercentRank,
    'StatefulPriceDistanceMA': StatefulPriceDistanceMA,
    'StatefulSRDistances': StatefulSRDistances,
}

class DualHorizonTradingEnvironment(gym.Env):
    """
    ✅ NEW: Revolutionary dual-horizon trading environment.
    
    Key innovations:
    1. 20-second decision frequency for precise execution
    2. 3-minute reward evaluation horizon for strategic thinking
    3. Forward-looking promise system for future-oriented learning
    4. Backward-looking realized PnL tracking for immediate feedback
    5. Combined reward system balancing short and long-term objectives
    """
    
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None):
        super().__init__()
        
        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.dual_horizon_cfg = self.strat_cfg.dual_horizon
        self.normalizer = normalizer
        
        logger.info("--- Initializing DUAL-HORIZON Trading Environment ---")
        logger.info(f" -> Decision Frequency: Every 20 seconds")
        logger.info(f" -> Reward Horizon: {self.dual_horizon_cfg.reward_horizon_steps} steps ({self.dual_horizon_cfg.reward_horizon_steps * 20} seconds)")
        logger.info(f" -> Immediate Weight: {self.dual_horizon_cfg.immediate_reward_weight:.1%}")
        logger.info(f" -> Delayed Weight: {self.dual_horizon_cfg.delayed_reward_weight:.1%}")
        logger.info(f" -> Promise Analytics: {self.dual_horizon_cfg.enable_promise_analytics}")
        logger.info(f" -> Realized PnL Tracking: {self.dual_horizon_cfg.enable_realized_pnl_tracking}")
        
        try:
            # Initialize dual-horizon components
            self.dual_horizon_calculator = DualHorizonRewardCalculator(
                self.dual_horizon_cfg, 
                leverage=self.strat_cfg.leverage
            )
            
            self.risk_manager = EnhancedRiskManager(self.cfg, leverage=self.strat_cfg.leverage)
            
            # Data setup
            base_df = df_base_ohlc.set_index('timestamp')
            
            # Get all unique timeframes required
            feature_timeframes = {calc.timeframe for calc in self.strat_cfg.stateful_calculators}
            model_timeframes = set(k.value.split('_')[-1].replace('m','T').replace('h','H').replace('s','S').upper()
                                  for k in self.strat_cfg.lookback_periods.keys())
            all_required_freqs = model_timeframes.union(feature_timeframes)
            
            self.timeframes = {}
            logger.info("Resampling data for all required timeframes...")
            for freq in all_required_freqs:
                if freq in ['CONTEXT', 'PORTFOLIO_STATE', 'PRECOMPUTED_FEATURES']:
                    continue
                if freq not in self.timeframes:
                    agg_rules = {
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                        'volume': 'sum', 'volume_delta': 'sum', 'vwap': 'last'
                    }
                    valid_agg_rules = {k: v for k, v in agg_rules.items() if k in base_df.columns}
                    df_resampled = base_df.resample(freq).agg(valid_agg_rules).ffill()
                    self.timeframes[freq] = df_resampled.dropna()
            
            self.base_timestamps = self.timeframes[self.cfg.base_bar_timeframe.value].index
            self.max_step = len(self.base_timestamps) - self.dual_horizon_cfg.reward_horizon_steps - 2
            
            # Action space: [position_signal, position_size]
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            
            # Observation space setup
            obs_spaces = {}
            seq_len = self.strat_cfg.sequence_length
            for key, lookback in self.strat_cfg.lookback_periods.items():
                key_str = key.value
                if key_str.startswith('ohlcv_'):
                    shape = (seq_len, lookback, 5)
                elif key_str.startswith('ohlc_'):
                    shape = (seq_len, lookback, 4)
                elif key in [FeatureKeys.PORTFOLIO_STATE, FeatureKeys.CONTEXT, FeatureKeys.PRECOMPUTED_FEATURES]:
                    shape = (seq_len, lookback)
                else:
                    shape = (seq_len, lookback)
                obs_spaces[key_str] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
            
            self.observation_space = spaces.Dict(obs_spaces)
            self.observation_history = deque(maxlen=self.strat_cfg.sequence_length)
            
            # Initialize stateful features
            self._initialize_stateful_features()
            
            # Dual-horizon tracking
            self.position_history = []  # Track position changes for realized PnL
            self.previous_position = 0.0
            self.position_entry_step = 0
            self.position_entry_price = 0.0
            
            # Enhanced tracking
            self.portfolio_history = deque(maxlen=500)
            self.previous_portfolio_value = None
            self.episode_returns = []
            self.previous_action = None
            self.consecutive_losses = 0
            self.episode_peak_value = 0.0
            self.market_regime = "UNCERTAIN"
            self.volatility_estimate = 0.02
            
            # Performance tracking
            self.step_rewards = []
            self.dual_horizon_analytics = []
            self.trade_count = 0
            self.winning_trades = 0
            
            logger.info("Dual-horizon environment initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize dual-horizon environment: {e}", exc_info=True)
            raise
    
    def _initialize_stateful_features(self):
        """Initialize stateful feature calculators."""
        logger.info("Initializing stateful feature calculators from config...")
        
        self.feature_calculators: Dict[str, Any] = {}
        self.feature_histories: Dict[str, deque] = {}
        self.last_update_timestamps: Dict[str, pd.Timestamp] = {}
        
        # Dynamically create calculators from strategy config
        for calc_cfg in self.strat_cfg.stateful_calculators:
            if calc_cfg.class_name not in STATEFUL_CALCULATOR_MAP:
                raise ValueError(f"Unknown stateful calculator class: {calc_cfg.class_name}")
            
            calculator_class = STATEFUL_CALCULATOR_MAP[calc_cfg.class_name]
            self.feature_calculators[calc_cfg.name] = calculator_class(**calc_cfg.params)
            self.last_update_timestamps[calc_cfg.timeframe] = pd.Timestamp(0, tz='UTC')
        
        # Initialize history deques for all context features
        for key in self.strat_cfg.context_feature_keys:
            self.feature_histories[key] = deque(maxlen=self.cfg.get_required_warmup_period() + 200)
    
    def _warmup_features(self, warmup_steps: int):
        """Pre-calculate feature history up to simulation start point."""
        logger.info(f"Warming up stateful features for {warmup_steps} steps...")
        for i in tqdm(range(warmup_steps), desc="Warming up features", leave=False, ncols=100):
            self._update_stateful_features(i)
    
    def _update_stateful_features(self, step_index: int):
        """Update stateful calculators and populate feature history."""
        current_timestamp = self.base_timestamps[step_index]
        
        # Update calculator states when new bars are available
        for calc_cfg in self.strat_cfg.stateful_calculators:
            timeframe = calc_cfg.timeframe
            df_tf = self.timeframes[timeframe]
            
            try:
                latest_bar_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
                latest_bar_timestamp = df_tf.index[latest_bar_idx]
            except KeyError:
                continue
            
            if latest_bar_timestamp > self.last_update_timestamps[timeframe]:
                self.last_update_timestamps[timeframe] = latest_bar_timestamp
                new_data_point = df_tf[calc_cfg.source_col].iloc[latest_bar_idx]
                self.feature_calculators[calc_cfg.name].update(new_data_point)
        
        # Populate feature history deques
        for calc_cfg in self.strat_cfg.stateful_calculators:
            calculator = self.feature_calculators[calc_cfg.name]
            values = calculator.get()
            
            if isinstance(values, dict):
                for key in calc_cfg.output_keys:
                    default_val = 1.0 if 'dist' in key else 0.0
                    self.feature_histories[key].append(values.get(key, default_val))
            else:
                if len(calc_cfg.output_keys) == 1:
                    key = calc_cfg.output_keys[0]
                    self.feature_histories[key].append(values)
    
    def _get_current_context_features(self) -> np.ndarray:
        """Get context features from pre-calculated history."""
        final_vector = [
            self.feature_histories[key][-1] if self.feature_histories[key] else 0.0
            for key in self.strat_cfg.context_feature_keys
        ]
        return np.array(final_vector, dtype=np.float32)
    
    def _update_market_regime_and_volatility(self, step_index: int):
        """Update market regime detection and volatility estimates."""
        try:
            base_df = self.timeframes[self.cfg.base_bar_timeframe.value]
            
            if step_index >= 50:
                recent_prices = base_df['close'].iloc[max(0, step_index-50):step_index+1]
                returns = recent_prices.pct_change().dropna().values
                
                if len(returns) > 10:
                    self.volatility_estimate = np.std(returns) * np.sqrt(252)
                    self.risk_manager.volatility_buffer.append(self.volatility_estimate)
                    
                self.market_regime = self.risk_manager.update_market_regime(returns, self.volatility_estimate)
                
        except Exception as e:
            logger.warning(f"Error updating market regime: {e}")
    
    def _detect_realized_trade(self, current_position: float, current_price: float, 
                              current_step: int) -> Optional[RealizedTrade]:
        """Detect if a trade was closed and calculate realized PnL."""
        try:
            # Check if position was reduced or closed
            position_change = current_position - self.previous_position
            
            if abs(position_change) < 1e-6:  # No significant position change
                return None
            
            # If we're reducing an existing position, we have a realized trade
            if (self.previous_position > 0 and position_change < 0) or \
               (self.previous_position < 0 and position_change > 0):
                
                # Calculate realized PnL
                position_closed = min(abs(position_change), abs(self.previous_position))
                
                if self.previous_position > 0:  # Closing long position
                    realized_pnl = (current_price - self.position_entry_price) * position_closed
                else:  # Closing short position
                    realized_pnl = (self.position_entry_price - current_price) * position_closed
                
                # Apply leverage
                realized_pnl *= self.strat_cfg.leverage
                
                hold_duration = current_step - self.position_entry_step
                
                return RealizedTrade(
                    step=current_step,
                    entry_price=self.position_entry_price,
                    exit_price=current_price,
                    position_size=position_closed,
                    realized_pnl=realized_pnl,
                    hold_duration=hold_duration
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting realized trade: {e}")
            return None
    
    def reset(self, seed=None, options=None):
        """Enhanced reset with dual-horizon initialization."""
        try:
            super().reset(seed=seed)
            
            # Reset portfolio state
            self.balance = 1000000.0
            self.asset_held = 0.0
            self.used_margin = 0.0
            self.entry_price = 0.0
            
            # Reset dual-horizon tracking
            self.dual_horizon_calculator = DualHorizonRewardCalculator(
                self.dual_horizon_cfg,
                leverage=self.strat_cfg.leverage
            )
            
            self.position_history = []
            self.previous_position = 0.0
            self.position_entry_step = 0
            self.position_entry_price = 0.0
            
            # Reset enhanced tracking
            self.consecutive_losses = 0
            self.episode_peak_value = self.balance
            self.market_regime = "UNCERTAIN"
            self.volatility_estimate = 0.02
            self.trade_count = 0
            self.winning_trades = 0
            
            # Setup warmup
            warmup_period = self.cfg.get_required_warmup_period()
            self._initialize_stateful_features()
            self._warmup_features(warmup_period)
            
            self.current_step = warmup_period
            self.observation_history.clear()
            self.portfolio_history.clear()
            self.episode_returns.clear()
            self.step_rewards.clear()
            self.dual_horizon_analytics.clear()
            
            self.previous_portfolio_value = self.balance
            self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
            
            # Initialize observation history
            for i in range(self.strat_cfg.sequence_length):
                step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
                self._update_stateful_features(step_idx)
                self._update_market_regime_and_volatility(step_idx)
                self.observation_history.append(self._get_single_step_observation(step_idx))
            
            observation = self._get_observation_sequence()
            
            info = {
                'balance': self.balance,
                'asset_held': self.asset_held,
                'portfolio_value': self.balance,
                'market_regime': self.market_regime,
                'volatility_estimate': self.volatility_estimate,
                'dual_horizon': self.dual_horizon_calculator.get_analytics()
            }
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Error resetting dual-horizon environment: {e}", exc_info=True)
            raise
    
    def step(self, action: np.ndarray):
        """✅ NEW: Dual-horizon step function with revolutionary reward system."""
        try:
            # Update features and market state
            self._update_stateful_features(self.current_step)
            self._update_market_regime_and_volatility(self.current_step)
            
            current_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            initial_portfolio_value = self.balance + unrealized_pnl
            
            # Update episode peak
            if initial_portfolio_value > self.episode_peak_value:
                self.episode_peak_value = initial_portfolio_value
            
            # Risk management (liquidation check)
            position_notional = abs(self.asset_held) * current_price
            margin_ratio = (self.used_margin - unrealized_pnl) / position_notional if position_notional > 0 else float('inf')
            
            if margin_ratio < self.strat_cfg.maintenance_margin_rate:
                # Progressive liquidation
                margin_deficit = self.strat_cfg.maintenance_margin_rate - margin_ratio
                liquidation_factor = min(1.0, margin_deficit * 5)
                liquidation_amount = self.asset_held * liquidation_factor
                liquidation_value = liquidation_amount * current_price
                liquidation_cost = abs(liquidation_value) * (self.cfg.transaction_fee_pct + 0.001)
                
                self.asset_held -= liquidation_amount
                self.balance += liquidation_value - liquidation_cost
                
                new_position_notional = abs(self.asset_held) * current_price
                self.used_margin = new_position_notional / self.strat_cfg.leverage
                
                if liquidation_factor >= 1.0:  # Full liquidation
                    self.current_step += 1
                    truncated = self.current_step >= self.max_step
                    self.observation_history.append(self._get_single_step_observation(self.current_step))
                    observation = self._get_observation_sequence()
                    
                    info = {
                        'portfolio_value': self.balance,
                        'margin_ratio': 0.0,
                        'liquidation': True,
                        'dual_horizon': self.dual_horizon_calculator.get_analytics()
                    }
                    
                    return observation, -3.0, True, truncated, info
            
            # ✅ DUAL-HORIZON: Create promise for future evaluation
            promise_analytics = self.dual_horizon_calculator.get_analytics()
            
            # Calculate dynamic position sizing
            dynamic_limit = self.risk_manager.calculate_dynamic_position_limit(
                self.volatility_estimate, 
                initial_portfolio_value, 
                self.market_regime,
                promise_analytics
            )
            
            # Execute position changes
            action_signal = np.clip(action[0], -1.0, 1.0)
            action_size = np.clip(action[1], 0.0, 1.0)
            
            effective_size = action_size * dynamic_limit
            target_notional = initial_portfolio_value * action_signal * effective_size
            target_asset_quantity = target_notional / current_price if current_price > 1e-8 else 0
            
            # Risk checks
            max_allowable_margin = initial_portfolio_value * self.strat_cfg.max_margin_allocation_pct
            required_margin_for_target = (abs(target_asset_quantity) * current_price) / self.strat_cfg.leverage
            
            if required_margin_for_target > max_allowable_margin:
                capped_notional = max_allowable_margin * self.strat_cfg.leverage
                target_asset_quantity = (capped_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0
            
            required_margin = (abs(target_asset_quantity) * current_price) / self.strat_cfg.leverage
            if required_margin > initial_portfolio_value:
                max_affordable_notional = initial_portfolio_value * self.strat_cfg.leverage
                target_asset_quantity = (max_affordable_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0
            
            # ✅ DUAL-HORIZON: Detect realized trades BEFORE position change
            realized_trade = self._detect_realized_trade(target_asset_quantity, current_price, self.current_step)
            
            # Execute trade
            trade_quantity = target_asset_quantity - self.asset_held
            trade_notional = abs(trade_quantity) * current_price
            
            # Transaction costs
            base_fee = trade_notional * self.cfg.transaction_fee_pct
            slippage_cost = trade_notional * self.cfg.slippage_pct if abs(trade_quantity) > 0 else 0
            total_cost = base_fee + slippage_cost
            
            self.balance += unrealized_pnl - total_cost
            
            # Update position tracking
            if abs(trade_quantity) > 1e-8:
                self.previous_position = self.asset_held
                if self.asset_held == 0:  # Opening new position
                    self.position_entry_step = self.current_step
                    self.position_entry_price = current_price
            
            self.asset_held = target_asset_quantity
            new_notional_value = abs(self.asset_held) * current_price
            self.used_margin = new_notional_value / self.strat_cfg.leverage
            
            if abs(trade_quantity) > 1e-8:
                self.entry_price = current_price
                self.trade_count += 1
            
            # ✅ DUAL-HORIZON: Create new promise
            promise = self.dual_horizon_calculator.create_promise(
                self.current_step, action, current_price, initial_portfolio_value
            )
            
            # Move to next step
            self.current_step += 1
            truncated = self.current_step >= self.max_step
            
            # Calculate next portfolio value
            next_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
            next_portfolio_value = self.balance + next_unrealized_pnl
            
            # Termination conditions
            terminated = next_portfolio_value <= initial_portfolio_value * 0.5
            
            # ✅ DUAL-HORIZON: Calculate dual-horizon rewards
            
            # 1. Immediate reward from realized PnL (backward-looking)
            immediate_reward = self.dual_horizon_calculator.calculate_realized_pnl_reward(realized_trade)
            
            # 2. Delayed reward from promise fulfillment (forward-looking)
            delayed_reward = self.dual_horizon_calculator.evaluate_promises(self.current_step, next_price)
            
            # 3. Combine rewards according to dual-horizon weights
            total_reward = self.dual_horizon_calculator.calculate_total_reward(immediate_reward, delayed_reward)
            
            # Track performance
            if total_reward < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
                
            if total_reward > 0.1:
                self.winning_trades += 1
            
            # Update tracking
            self.previous_action = action
            self.portfolio_history.append(next_portfolio_value)
            self.step_rewards.append(total_reward)
            
            # Store dual-horizon analytics
            analytics = self.dual_horizon_calculator.get_analytics()
            analytics.update({
                'step': self.current_step,
                'immediate_reward': immediate_reward,
                'delayed_reward': delayed_reward,
                'total_reward': total_reward,
                'realized_trade': realized_trade is not None,
                'promise_created': promise is not None
            })
            self.dual_horizon_analytics.append(analytics)
            
            if self.previous_portfolio_value is not None:
                period_return = (next_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
                self.episode_returns.append(period_return)
                self.risk_manager.return_buffer.append(period_return)
            
            self.previous_portfolio_value = next_portfolio_value
            
            self.observation_history.append(self._get_single_step_observation(self.current_step))
            observation = self._get_observation_sequence()
            
            # Enhanced info dictionary with dual-horizon data
            info = {
                'balance': self.balance,
                'asset_held': self.asset_held,
                'portfolio_value': next_portfolio_value,
                'drawdown': (self.episode_peak_value - next_portfolio_value) / self.episode_peak_value,
                'volatility': self.volatility_estimate,
                'unrealized_pnl': next_unrealized_pnl,
                'margin_ratio': margin_ratio,
                'used_margin': self.used_margin,
                'market_regime': self.market_regime,
                'consecutive_losses': self.consecutive_losses,
                'trade_count': self.trade_count,
                'win_rate': self.winning_trades / max(self.trade_count, 1),
                'transaction_cost': total_cost,
                
                # ✅ DUAL-HORIZON: New info fields
                'dual_horizon': analytics,
                'immediate_reward': immediate_reward,
                'delayed_reward': delayed_reward,
                'promise_created': promise is not None,
                'realized_trade': realized_trade is not None,
                'reward_horizon_steps': self.dual_horizon_cfg.reward_horizon_steps
            }
            
            return observation, total_reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in dual-horizon environment step: {e}", exc_info=True)
            observation = self._get_observation_sequence()
            info = {
                'portfolio_value': self.balance, 
                'error': True,
                'dual_horizon': self.dual_horizon_calculator.get_analytics()
            }
            return observation, -2.0, True, False, info
    
    def _get_single_step_observation(self, step_index) -> dict:
        """Get single step observation with enhanced portfolio state."""
        try:
            if self.normalizer is None:
                return {}
            
            current_timestamp = self.base_timestamps[step_index]
            raw_obs = {}
            
            base_df = self.timeframes[self.cfg.base_bar_timeframe.value]
            current_price = base_df['close'].iloc[base_df.index.get_loc(current_timestamp, method='ffill')]
            
            # Process each observation type
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.CONTEXT.value, FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.PRECOMPUTED_FEATURES.value]:
                    continue
                
                freq = key.split('_')[-1].replace('m','T').replace('h','H').replace('s', 'S').upper()
                df_tf = self.timeframes[freq]
                
                end_idx = df_tf.index.get_loc(current_timestamp, method='ffill')
                start_idx = max(0, end_idx - lookback + 1)
                
                if key.startswith('price_'):
                    window = df_tf.iloc[start_idx : end_idx + 1]['close'].values.astype(np.float32)
                    if len(window) < lookback:
                        window = np.pad(window, (lookback - len(window), 0), 'edge')
                    raw_obs[key] = window
                
                elif key.startswith('volume_delta_'):
                    window = df_tf.iloc[start_idx : end_idx + 1]['volume_delta'].values.astype(np.float32)
                    if len(window) < lookback:
                        window = np.pad(window, (lookback - len(window), 0), 'constant')
                    raw_obs[key] = window
                
                elif key.startswith('ohlc'):
                    cols = ['open','high','low','close','volume'] if 'v' in key else ['open','high','low','close']
                    window = df_tf.iloc[start_idx : end_idx + 1][cols].values.astype(np.float32)
                    if len(window) < lookback:
                        padding = np.repeat(window[0:1], lookback - len(window), axis=0)
                        window = np.concatenate([padding, window], axis=0)
                    raw_obs[key] = window
            
            # Context features
            raw_obs[FeatureKeys.CONTEXT.value] = self._get_current_context_features()
            
            # Precomputed features
            current_bar_features = base_df.loc[base_df.index.get_loc(current_timestamp, method='ffill')]
            precomputed_vector = current_bar_features[self.strat_cfg.precomputed_feature_keys].fillna(0.0).values.astype(np.float32)
            raw_obs[FeatureKeys.PRECOMPUTED_FEATURES.value] = precomputed_vector
            
            # ✅ DUAL-HORIZON: Enhanced portfolio state with dual-horizon information
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            portfolio_value = self.balance + unrealized_pnl
            current_notional = self.asset_held * current_price
            
            # Standard portfolio features
            normalized_position = np.clip(current_notional / (portfolio_value + 1e-9), -1.0, 1.0)
            pnl_on_margin = unrealized_pnl / (self.used_margin + 1e-9)
            normalized_pnl = np.tanh(pnl_on_margin)
            
            position_notional = abs(self.asset_held) * current_price
            margin_health = self.used_margin + unrealized_pnl
            margin_ratio = np.clip(margin_health / position_notional, 0, 2.0) if position_notional > 0 else 2.0
            
            # Market regime encoding
            regime_encoding = 0.0
            if self.market_regime == "HIGH_VOLATILITY":
                regime_encoding = 1.0
            elif self.market_regime == "TRENDING_UP":
                regime_encoding = 0.8
            elif self.market_regime == "TRENDING_DOWN":
                regime_encoding = -0.8
            elif self.market_regime == "LOW_VOLATILITY":
                regime_encoding = 0.6
            elif self.market_regime == "SIDEWAYS":
                regime_encoding = 0.0
            else:  # UNCERTAIN
                regime_encoding = -0.2
            
            # ✅ NEW: Dual-horizon awareness signal
            dual_horizon_analytics = self.dual_horizon_calculator.get_analytics()
            promise_efficiency = dual_horizon_analytics.get('promise_success_rate', 0.5)
            promise_efficiency_normalized = (promise_efficiency - 0.5) * 2  # Scale to [-1, 1]
            
            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([
                normalized_position,
                normalized_pnl,
                margin_ratio,
                regime_encoding,
                promise_efficiency_normalized  # ✅ NEW: Promise performance feedback
            ], dtype=np.float32)
            
            return self.normalizer.transform(raw_obs)
            
        except Exception as e:
            logger.error(f"Error getting dual-horizon observation for step {step_index}: {e}", exc_info=True)
            
            # Return safe fallback observation
            obs = {}
            for key_enum, lookback in self.strat_cfg.lookback_periods.items():
                key = key_enum.value
                if key in [FeatureKeys.PORTFOLIO_STATE.value, FeatureKeys.CONTEXT.value, FeatureKeys.PRECOMPUTED_FEATURES.value]:
                    obs[key] = np.zeros(lookback, dtype=np.float32)
                elif key.startswith('ohlcv_'):
                    obs[key] = np.zeros((lookback, 5), dtype=np.float32)
                elif key.startswith('ohlc_'):
                    obs[key] = np.zeros((lookback, 4), dtype=np.float32)
                else:
                    obs[key] = np.zeros(lookback, dtype=np.float32)
            return obs
    
    def _get_observation_sequence(self):
        """Get observation sequence with enhanced error handling."""
        try:
            return {
                key: np.stack([obs[key] for obs in self.observation_history])
                for key in self.observation_space.spaces.keys()
            }
        except Exception as e:
            logger.error(f"Error getting observation sequence: {e}")
            return {
                key: np.zeros(self.observation_space.spaces[key].shape, dtype=np.float32)
                for key in self.observation_space.spaces.keys()
            }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics including dual-horizon analytics."""
        try:
            if len(self.portfolio_history) < 2:
                return {}
            
            portfolio_values = np.array(self.portfolio_history)
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            
            # Basic metrics
            total_return = (final_value - initial_value) / initial_value
            
            # Risk metrics
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
            
            # Drawdown metrics
            cumulative_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (cumulative_max - portfolio_values) / cumulative_max
            max_drawdown = np.max(drawdowns)
            
            # Risk-adjusted returns
            excess_return = total_return - 0.02
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
            
            # Trading metrics
            win_rate = self.winning_trades / max(self.trade_count, 1)
            avg_reward = np.mean(self.step_rewards) if self.step_rewards else 0.0
            reward_volatility = np.std(self.step_rewards) if len(self.step_rewards) > 1 else 0.0
            
            # ✅ DUAL-HORIZON: Add dual-horizon specific metrics
            dual_horizon_analytics = self.dual_horizon_calculator.get_analytics()
            
            base_metrics = {
                'total_return': total_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': self.trade_count,
                'avg_reward': avg_reward,
                'reward_volatility': reward_volatility,
                'consecutive_losses': self.consecutive_losses,
                'final_portfolio_value': final_value,
            }
            
            # Merge with dual-horizon analytics
            base_metrics.update(dual_horizon_analytics)
            
            return base_metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'dual_horizon': self.dual_horizon_calculator.get_analytics()}

# --- CONVENIENCE FUNCTIONS ---

def create_dual_horizon_trading_environment(df_base_ohlc: pd.DataFrame, normalizer: Normalizer, config=None):
    """
    Convenience function to create dual-horizon trading environment.
    
    Args:
        df_base_ohlc: Base OHLC data
        normalizer: Fitted normalizer
        config: Configuration object (uses SETTINGS if None)
    """
    return DualHorizonTradingEnvironment(
        df_base_ohlc=df_base_ohlc,
        normalizer=normalizer,
        config=config
    )

if __name__ == "__main__":
    # Example usage and testing
    try:
        logger.info("Testing dual-horizon trading environment...")
        
        # Test the dual-horizon reward calculator
        dual_config = DualHorizonConfig()
        reward_calc = DualHorizonRewardCalculator(dual_config, leverage=10.0)
        
        # Test promise creation
        test_action = np.array([0.5, 0.3])
        promise = reward_calc.create_promise(0, test_action, 50000.0, 1000000.0)
        
        if promise:
            logger.info(f"Created promise: {promise}")
        
        # Test promise evaluation
        delayed_reward = reward_calc.evaluate_promises(9, 50100.0)  # 9 steps later, price up 0.2%
        logger.info(f"Delayed reward: {delayed_reward}")
        
        # Test realized trade reward
        realized_trade = RealizedTrade(
            step=10,
            entry_price=50000.0,
            exit_price=50200.0,
            position_size=1.0,
            realized_pnl=200.0,
            hold_duration=5
        )
        
        immediate_reward = reward_calc.calculate_realized_pnl_reward(realized_trade)
        logger.info(f"Immediate reward: {immediate_reward}")
        
        # Test combined reward
        total_reward = reward_calc.calculate_total_reward(immediate_reward, delayed_reward)
        logger.info(f"Total dual-horizon reward: {total_reward}")
        
        # Test analytics
        analytics = reward_calc.get_analytics()
        logger.info(f"Dual-horizon analytics: {analytics}")
        
        logger.info("✅ Dual-horizon trading environment test completed!")
        
    except Exception as e:
        logger.error(f"Dual-horizon environment test failed: {e}")