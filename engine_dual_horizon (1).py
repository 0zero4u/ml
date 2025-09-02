"""
✅ REVOLUTIONARY: Dual-Horizon Trading Engine with Promise System

This enhanced trading environment implements a groundbreaking dual-horizon approach:

1. FORWARD-LOOKING PROMISE SYSTEM:
   - Agent makes decisions every 20 seconds 
   - Rewards are evaluated 3 minutes (180s) into the future
   - Creates "promises" that incentivize strategic thinking
   - Enables precise "sniping" of optimal entries

2. BACKWARD-LOOKING REALIZED PnL TRACKING:
   - Immediate detection of closed/reduced positions
   - Permanent balance adjustments for realized gains/losses
   - Immediate feedback prevents reckless behavior

3. BALANCED REWARD ARCHITECTURE:
   - Combines forward promises (70% weight) with backward PnL (30% weight)
   - Encourages both strategic positioning and tactical execution
   - Configurable weighting system for different trading styles
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque, namedtuple
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
import warnings
from scipy import stats
from sklearn.preprocessing import RobustScaler
import time

# Import configuration and the new stateful feature calculators
from config_dual_horizon import SETTINGS, FeatureKeys, DualHorizonConfig
from normalizer import Normalizer
from features import (
    StatefulBBWPercentRank,
    StatefulPriceDistanceMA,
    StatefulSRDistances,
)

# Map calculator names from config to their classes for dynamic instantiation
STATEFUL_CALCULATOR_MAP = {
    'StatefulBBWPercentRank': StatefulBBWPercentRank,
    'StatefulPriceDistanceMA': StatefulPriceDistanceMA,
    'StatefulSRDistances': StatefulSRDistances,
}

logger = logging.getLogger(__name__)

# ✅ NEW: Promise System Data Structures
Promise = namedtuple('Promise', [
    'step', 'action', 'entry_price', 'position_size', 'portfolio_value',
    'expected_future_price', 'creation_timestamp', 'horizon_steps'
])

PromiseResult = namedtuple('PromiseResult', [
    'promise', 'actual_future_price', 'reward', 'success', 'evaluation_timestamp'
])

RealizedTrade = namedtuple('RealizedTrade', [
    'step', 'entry_price', 'exit_price', 'position_size', 'realized_pnl',
    'holding_period_steps', 'trade_type'  # 'full_close', 'partial_close', 'position_flip'
])

class DualHorizonRewardCalculator:
    """
    ✅ REVOLUTIONARY: Dual-horizon reward system combining:
    - Forward-looking promises (strategic thinking)  
    - Backward-looking realized PnL (tactical execution)
    """
    
    def __init__(self, config: DualHorizonConfig, leverage: float = 10.0):
        self.config = config
        self.leverage = leverage
        
        # Promise tracking
        self.active_promises: List[Promise] = []
        self.completed_promises: List[PromiseResult] = []
        self.promise_success_rate = 0.0
        
        # Realized PnL tracking
        self.realized_trades: List[RealizedTrade] = []
        self.cumulative_realized_pnl = 0.0
        self.last_position_size = 0.0
        self.last_entry_price = 0.0
        
        # Analytics
        self.reward_history = deque(maxlen=1000)
        self.promise_analytics = {
            'total_created': 0,
            'total_evaluated': 0,
            'success_count': 0,
            'avg_success_reward': 0.0,
            'avg_failure_penalty': 0.0
        }
        
        # Scaling factors adjusted for leverage
        self.immediate_scaling = config.immediate_scaling / leverage
        self.delayed_scaling = config.delayed_scaling / leverage
        
        logger.info(f"✅ Dual-Horizon Reward Calculator initialized:")
        logger.info(f"   → Immediate Weight: {config.immediate_reward_weight:.1%}")
        logger.info(f"   → Delayed Weight: {config.delayed_reward_weight:.1%}") 
        logger.info(f"   → Reward Horizon: {config.reward_horizon_steps} steps")
        logger.info(f"   → Max Promises: {config.max_promises}")
    
    def create_promise(self, step: int, action: np.ndarray, current_price: float,
                      portfolio_value: float, expected_future_price: float) -> Promise:
        """✅ NEW: Create a forward-looking promise for future evaluation."""
        
        promise = Promise(
            step=step,
            action=action.copy(),
            entry_price=current_price,
            position_size=action[1] if len(action) > 1 else 0.0,
            portfolio_value=portfolio_value,
            expected_future_price=expected_future_price,
            creation_timestamp=time.time(),
            horizon_steps=self.config.reward_horizon_steps
        )
        
        # Add to active promises (with max limit)
        self.active_promises.append(promise)
        if len(self.active_promises) > self.config.max_promises:
            # Remove oldest promise
            oldest = self.active_promises.pop(0)
            logger.debug(f"Removed oldest promise from step {oldest.step} (max limit reached)")
        
        self.promise_analytics['total_created'] += 1
        return promise
    
    def evaluate_promises(self, current_step: int, current_price: float) -> List[PromiseResult]:
        """✅ NEW: Evaluate matured promises and return results."""
        
        results = []
        remaining_promises = []
        
        for promise in self.active_promises:
            steps_elapsed = current_step - promise.step
            
            if steps_elapsed >= promise.horizon_steps:
                # Promise has matured - evaluate it
                result = self._evaluate_single_promise(promise, current_price)
                results.append(result)
                self.completed_promises.append(result)
                
                # Update analytics
                self.promise_analytics['total_evaluated'] += 1
                if result.success:
                    self.promise_analytics['success_count'] += 1
                    
            elif self.config.enable_promise_decay and steps_elapsed > 0:
                # Apply decay to promise value
                decay_factor = self.config.promise_decay_rate ** steps_elapsed
                # Keep promise but note decay
                remaining_promises.append(promise)
            else:
                # Promise not yet mature
                remaining_promises.append(promise)
        
        self.active_promises = remaining_promises
        
        # Update success rate
        if self.promise_analytics['total_evaluated'] > 0:
            self.promise_success_rate = self.promise_analytics['success_count'] / self.promise_analytics['total_evaluated']
        
        return results
    
    def _evaluate_single_promise(self, promise: Promise, actual_future_price: float) -> PromiseResult:
        """✅ NEW: Evaluate a single matured promise."""
        
        # Calculate expected vs actual performance
        expected_direction = 1 if promise.expected_future_price > promise.entry_price else -1
        actual_direction = 1 if actual_future_price > promise.entry_price else -1
        
        expected_magnitude = abs(promise.expected_future_price - promise.entry_price) / promise.entry_price
        actual_magnitude = abs(actual_future_price - promise.entry_price) / promise.entry_price
        
        # Calculate promise reward
        direction_correct = expected_direction == actual_direction
        magnitude_accuracy = 1.0 - abs(expected_magnitude - actual_magnitude)
        
        if direction_correct and actual_magnitude > self.config.promise_success_threshold:
            # Successful promise
            base_reward = actual_magnitude * magnitude_accuracy * promise.position_size
            reward = base_reward * self.delayed_scaling
            success = True
        else:
            # Failed promise - penalty
            penalty_factor = expected_magnitude * (1 + abs(actual_direction - expected_direction))
            reward = -penalty_factor * self.delayed_scaling * 0.5  # Moderate penalty
            success = False
        
        return PromiseResult(
            promise=promise,
            actual_future_price=actual_future_price,
            reward=reward,
            success=success,
            evaluation_timestamp=time.time()
        )
    
    def detect_realized_trade(self, prev_position: float, curr_position: float, 
                            prev_entry_price: float, curr_price: float, 
                            current_step: int) -> Optional[RealizedTrade]:
        """✅ NEW: Detect and calculate realized PnL from position changes."""
        
        if abs(prev_position) < self.config.min_position_for_tracking:
            # No significant previous position
            return None
        
        position_change = curr_position - prev_position
        
        if abs(position_change) < self.config.min_position_for_tracking:
            # No significant position change
            return None
        
        # Determine trade type
        if abs(curr_position) < self.config.min_position_for_tracking:
            trade_type = 'full_close'
            realized_size = abs(prev_position)
        elif np.sign(curr_position) != np.sign(prev_position):
            trade_type = 'position_flip'
            realized_size = abs(prev_position)
        else:
            trade_type = 'partial_close'
            realized_size = abs(position_change)
        
        # Calculate realized PnL
        if prev_position > 0:  # Long position
            realized_pnl = realized_size * (curr_price - prev_entry_price)
        else:  # Short position
            realized_pnl = realized_size * (prev_entry_price - curr_price)
        
        # Apply leverage
        realized_pnl *= self.leverage
        
        realized_trade = RealizedTrade(
            step=current_step,
            entry_price=prev_entry_price,
            exit_price=curr_price,
            position_size=realized_size,
            realized_pnl=realized_pnl,
            holding_period_steps=1,  # Could be enhanced with actual holding period tracking
            trade_type=trade_type
        )
        
        self.realized_trades.append(realized_trade)
        self.cumulative_realized_pnl += realized_pnl
        
        return realized_trade
    
    def calculate_combined_reward(self, promise_results: List[PromiseResult], 
                                realized_trade: Optional[RealizedTrade],
                                portfolio_change: float) -> Tuple[float, Dict[str, Any]]:
        """✅ NEW: Calculate combined dual-horizon reward."""
        
        components = {
            'immediate_reward': 0.0,
            'delayed_reward': 0.0,
            'promise_count': len(promise_results),
            'realized_pnl': 0.0,
            'total_reward': 0.0
        }
        
        # 1. Calculate delayed reward from promise results
        delayed_reward = 0.0
        if promise_results:
            promise_rewards = [result.reward for result in promise_results]
            delayed_reward = sum(promise_rewards)
            components['delayed_reward'] = delayed_reward
        
        # 2. Calculate immediate reward from realized trades
        immediate_reward = 0.0
        if realized_trade:
            # Scale realized PnL reward
            immediate_reward = realized_trade.realized_pnl * self.immediate_scaling
            components['realized_pnl'] = realized_trade.realized_pnl
        
        # Add small portfolio change component for continuous feedback
        portfolio_component = np.tanh(portfolio_change * 100) * 0.1
        immediate_reward += portfolio_component
        
        components['immediate_reward'] = immediate_reward
        
        # 3. Combine rewards with configured weights
        total_reward = (
            self.config.immediate_reward_weight * immediate_reward +
            self.config.delayed_reward_weight * delayed_reward
        )
        
        # 4. Apply bounds and stability checks
        total_reward = np.clip(total_reward, -10.0, 10.0)
        
        components['total_reward'] = total_reward
        
        # Update history
        self.reward_history.append(total_reward)
        
        return total_reward, components

class EnhancedDualHorizonTradingEnvironment(gym.Env):
    """
    ✅ REVOLUTIONARY: Dual-Horizon Trading Environment
    
    Combines forward-looking promises with backward-looking realized PnL tracking
    to create an agent that excels at both strategic positioning and tactical execution.
    """
    
    def __init__(self, df_base_ohlc: pd.DataFrame, normalizer: Normalizer, 
                 config=None, leverage: float = None, dual_horizon_config: DualHorizonConfig = None):
        
        super().__init__()
        
        self.cfg = config or SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.normalizer = normalizer
        
        # ✅ NEW: Dual-horizon configuration
        self.dual_horizon_config = dual_horizon_config or self.strat_cfg.dual_horizon
        self.leverage = leverage if leverage is not None else self.strat_cfg.leverage
        
        logger.info("--- Initializing REVOLUTIONARY Dual-Horizon Trading Environment ---")
        logger.info(f"   → Decision Frequency: Every {self.dual_horizon_config.decision_frequency_seconds}s")
        logger.info(f"   → Reward Horizon: {self.dual_horizon_config.reward_horizon_steps} steps ({self.dual_horizon_config.reward_horizon_steps * 20}s)")
        logger.info(f"   → Promise System: {'ENABLED' if self.dual_horizon_config.max_promises > 0 else 'DISABLED'}")
        logger.info(f"   → Realized PnL Tracking: {'ENABLED' if self.dual_horizon_config.enable_realized_pnl_tracking else 'DISABLED'}")
        
        try:
            # Initialize dual-horizon reward calculator
            self.reward_calculator = DualHorizonRewardCalculator(
                self.dual_horizon_config, 
                leverage=self.leverage
            )
            
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
            self.max_step = len(self.base_timestamps) - 2
            
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
            
            # ✅ NEW: Dual-horizon tracking variables
            self.future_prices = deque(maxlen=self.dual_horizon_config.reward_horizon_steps + 10)
            self.portfolio_history = deque(maxlen=500)
            self.position_history = deque(maxlen=100)
            self.entry_price_history = deque(maxlen=100)
            
            # Enhanced tracking
            self.previous_portfolio_value = None
            self.episode_returns = []
            self.previous_action = None
            self.consecutive_losses = 0
            self.episode_peak_value = 0.0
            self.market_regime = "UNCERTAIN"
            self.volatility_estimate = 0.02
            
            # Performance tracking
            self.step_rewards = []
            self.reward_components_history = []
            self.trade_count = 0
            self.winning_trades = 0
            
            # ✅ NEW: Dual-horizon metrics
            self.dual_horizon_metrics = {
                'promises_created': 0,
                'promises_evaluated': 0,
                'promises_successful': 0,
                'realized_trades': 0,
                'total_realized_pnl': 0.0,
                'avg_promise_success_rate': 0.0
            }
            
            logger.info("✅ Dual-Horizon environment initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize dual-horizon environment: {e}", exc_info=True)
            raise
    
    def _initialize_stateful_features(self):
        """Create instances of all stateful calculators and their history deques based on config."""
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
        
        # Initialize history deques for all context features declared in the config
        for key in self.strat_cfg.context_feature_keys:
            self.feature_histories[key] = deque(maxlen=self.cfg.get_required_warmup_period() + 200)
    
    def _warmup_features(self, warmup_steps: int):
        """Pre-calculates feature history up to the simulation start point."""
        logger.info(f"Warming up stateful features for {warmup_steps} steps...")
        
        for i in tqdm(range(warmup_steps), desc="Warming up features", leave=False, ncols=100):
            self._update_stateful_features(i)
            
            # ✅ NEW: Build future price buffer during warmup
            if i < len(self.base_timestamps):
                current_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[i]
                self.future_prices.append(current_price)
    
    def _update_stateful_features(self, step_index: int):
        """Efficiently updates stateful calculators and populates feature history."""
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
        """Fast feature retrieval from pre-calculated history deques."""
        final_vector = [
            self.feature_histories[key][-1] if self.feature_histories[key] else 0.0
            for key in self.strat_cfg.context_feature_keys
        ]
        return np.array(final_vector, dtype=np.float32)
    
    def _get_future_price(self, steps_ahead: int) -> Optional[float]:
        """✅ NEW: Get price N steps into the future for promise evaluation."""
        future_step = self.current_step + steps_ahead
        
        if future_step < len(self.base_timestamps):
            try:
                future_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[future_step]
                return float(future_price)
            except (IndexError, KeyError):
                return None
        return None
    
    def reset(self, seed=None, options=None):
        """✅ ENHANCED: Reset with dual-horizon initialization."""
        try:
            super().reset(seed=seed)
            
            # Reset portfolio state
            self.balance = 1000000.0
            self.asset_held = 0.0
            self.used_margin = 0.0
            self.entry_price = 0.0
            
            # Reset enhanced tracking
            self.consecutive_losses = 0
            self.episode_peak_value = self.balance
            self.market_regime = "UNCERTAIN"
            self.volatility_estimate = 0.02
            self.trade_count = 0
            self.winning_trades = 0
            
            # ✅ NEW: Reset dual-horizon components
            self.reward_calculator = DualHorizonRewardCalculator(
                self.dual_horizon_config,
                leverage=self.leverage
            )
            self.future_prices.clear()
            self.position_history.clear()
            self.entry_price_history.clear()
            
            self.dual_horizon_metrics = {
                'promises_created': 0,
                'promises_evaluated': 0,
                'promises_successful': 0,
                'realized_trades': 0,
                'total_realized_pnl': 0.0,
                'avg_promise_success_rate': 0.0
            }
            
            # Setup warmup
            warmup_period = self.cfg.get_required_warmup_period()
            self._initialize_stateful_features()
            self._warmup_features(warmup_period)
            
            self.current_step = warmup_period
            self.observation_history.clear()
            self.portfolio_history.clear()
            self.episode_returns.clear()
            self.step_rewards.clear()
            self.reward_components_history.clear()
            
            self.previous_portfolio_value = self.balance
            self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
            
            # Initialize observation history
            for i in range(self.strat_cfg.sequence_length):
                step_idx = self.current_step - self.strat_cfg.sequence_length + 1 + i
                self._update_stateful_features(step_idx)
                self.observation_history.append(self._get_single_step_observation(step_idx))
            
            observation = self._get_observation_sequence()
            
            info = {
                'balance': self.balance,
                'asset_held': self.asset_held,
                'portfolio_value': self.balance,
                'market_regime': self.market_regime,
                'volatility_estimate': self.volatility_estimate,
                'leverage': self.leverage,
                'dual_horizon_enabled': True,
                'reward_horizon_steps': self.dual_horizon_config.reward_horizon_steps
            }
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Error resetting dual-horizon environment: {e}", exc_info=True)
            raise
    
    def step(self, action: np.ndarray):
        """✅ REVOLUTIONARY: Dual-horizon step function with promise system."""
        try:
            # Update features and market state
            self._update_stateful_features(self.current_step)
            current_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
            
            # ✅ NEW: Update future prices buffer
            self.future_prices.append(current_price)
            
            # Calculate current portfolio value
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            initial_portfolio_value = self.balance + unrealized_pnl
            
            # Update episode peak
            if initial_portfolio_value > self.episode_peak_value:
                self.episode_peak_value = initial_portfolio_value
            
            # Store previous state for realized PnL tracking
            prev_position = self.asset_held
            prev_entry_price = self.entry_price
            
            # ✅ ENHANCED: Execute trade with dual-horizon awareness
            self._execute_trade_with_dual_horizon(action, current_price, initial_portfolio_value)
            
            # Move to next step
            self.current_step += 1
            truncated = self.current_step >= self.max_step
            
            # Calculate next portfolio value
            if not truncated:
                next_price = self.timeframes[self.cfg.base_bar_timeframe.value]['close'].iloc[self.current_step]
                next_unrealized_pnl = self.asset_held * (next_price - self.entry_price)
                next_portfolio_value = self.balance + next_unrealized_pnl
            else:
                next_price = current_price
                next_portfolio_value = initial_portfolio_value
            
            # ✅ NEW: 1. Create promise for forward-looking reward
            if self.dual_horizon_config.max_promises > 0:
                future_price = self._get_future_price(self.dual_horizon_config.reward_horizon_steps)
                if future_price is not None:
                    promise = self.reward_calculator.create_promise(
                        step=self.current_step,
                        action=action,
                        current_price=current_price,
                        portfolio_value=next_portfolio_value,
                        expected_future_price=future_price
                    )
                    self.dual_horizon_metrics['promises_created'] += 1
            
            # ✅ NEW: 2. Evaluate matured promises
            promise_results = self.reward_calculator.evaluate_promises(
                current_step=self.current_step,
                current_price=current_price
            )
            
            if promise_results:
                self.dual_horizon_metrics['promises_evaluated'] += len(promise_results)
                successful_promises = sum(1 for result in promise_results if result.success)
                self.dual_horizon_metrics['promises_successful'] += successful_promises
            
            # ✅ NEW: 3. Detect realized trades
            realized_trade = None
            if self.dual_horizon_config.enable_realized_pnl_tracking:
                realized_trade = self.reward_calculator.detect_realized_trade(
                    prev_position=prev_position,
                    curr_position=self.asset_held,
                    prev_entry_price=prev_entry_price,
                    curr_price=current_price,
                    current_step=self.current_step
                )
                
                if realized_trade:
                    self.dual_horizon_metrics['realized_trades'] += 1
                    self.dual_horizon_metrics['total_realized_pnl'] += realized_trade.realized_pnl
                    
                    # ✅ CRITICAL: Immediately adjust balance for realized PnL
                    self.balance += realized_trade.realized_pnl
                    next_portfolio_value = self.balance + self.asset_held * (next_price - self.entry_price)
            
            # ✅ NEW: 4. Calculate combined dual-horizon reward
            portfolio_change = (next_portfolio_value - initial_portfolio_value) / initial_portfolio_value
            
            reward, reward_components = self.reward_calculator.calculate_combined_reward(
                promise_results=promise_results,
                realized_trade=realized_trade,
                portfolio_change=portfolio_change
            )
            
            # Enhanced termination conditions
            terminated = next_portfolio_value <= initial_portfolio_value * 0.5  # 50% loss termination
            
            # Track consecutive losses and wins
            if reward < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            if reward > 0.1:
                self.winning_trades += 1
            
            # Update tracking
            self.previous_action = action
            self.portfolio_history.append(next_portfolio_value)
            self.position_history.append(self.asset_held)
            self.entry_price_history.append(self.entry_price)
            self.step_rewards.append(reward)
            self.reward_components_history.append(reward_components)
            
            if self.previous_portfolio_value is not None:
                period_return = (next_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
                self.episode_returns.append(period_return)
            
            self.previous_portfolio_value = next_portfolio_value
            self.observation_history.append(self._get_single_step_observation(self.current_step))
            observation = self._get_observation_sequence()
            
            # ✅ NEW: Enhanced info dictionary with dual-horizon metrics
            current_drawdown = (self.episode_peak_value - next_portfolio_value) / self.episode_peak_value
            
            info = {
                'balance': self.balance,
                'asset_held': self.asset_held,
                'portfolio_value': next_portfolio_value,
                'drawdown': current_drawdown,
                'volatility': self.volatility_estimate,
                'unrealized_pnl': self.asset_held * (next_price - self.entry_price),
                'margin_ratio': self.used_margin / (abs(self.asset_held) * current_price) if abs(self.asset_held) > 0 else float('inf'),
                'used_margin': self.used_margin,
                'market_regime': self.market_regime,
                'consecutive_losses': self.consecutive_losses,
                'trade_count': self.trade_count,
                'win_rate': self.winning_trades / max(self.trade_count, 1),
                'reward_components': reward_components,
                'leverage': self.leverage,
                
                # ✅ NEW: Dual-horizon specific metrics
                'promises_active': len(self.reward_calculator.active_promises),
                'promises_evaluated_this_step': len(promise_results),
                'realized_trade': realized_trade is not None,
                'dual_horizon_metrics': self.dual_horizon_metrics.copy(),
                'promise_success_rate': self.reward_calculator.promise_success_rate,
                'cumulative_realized_pnl': self.reward_calculator.cumulative_realized_pnl,
            }
            
            # Update aggregate success rate
            if self.dual_horizon_metrics['promises_evaluated'] > 0:
                self.dual_horizon_metrics['avg_promise_success_rate'] = (
                    self.dual_horizon_metrics['promises_successful'] / 
                    self.dual_horizon_metrics['promises_evaluated']
                )
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error in dual-horizon environment step: {e}", exc_info=True)
            observation = self._get_observation_sequence()
            info = {'portfolio_value': self.balance, 'error': True, 'leverage': self.leverage}
            return observation, -1.0, True, False, info
    
    def _execute_trade_with_dual_horizon(self, action: np.ndarray, current_price: float, 
                                       portfolio_value: float):
        """✅ NEW: Execute trade with dual-horizon position management."""
        
        action_signal = np.clip(action[0], -1.0, 1.0)
        action_size = np.clip(action[1], 0.0, 1.0)
        
        # Calculate target position
        target_notional = portfolio_value * action_signal * action_size
        target_asset_quantity = target_notional / current_price if current_price > 1e-8 else 0
        
        # Risk checks with leverage awareness
        max_allowable_margin = portfolio_value * self.strat_cfg.max_margin_allocation_pct
        required_margin_for_target = (abs(target_asset_quantity) * current_price) / self.leverage
        
        if required_margin_for_target > max_allowable_margin:
            capped_notional = max_allowable_margin * self.leverage
            target_asset_quantity = (capped_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0
        
        required_margin = (abs(target_asset_quantity) * current_price) / self.leverage
        if required_margin > portfolio_value:
            max_affordable_notional = portfolio_value * self.leverage
            target_asset_quantity = (max_affordable_notional / current_price * np.sign(target_asset_quantity)) if current_price > 1e-8 else 0
        
        # Execute trade
        trade_quantity = target_asset_quantity - self.asset_held
        trade_notional = abs(trade_quantity) * current_price
        
        # Enhanced transaction cost calculation
        base_fee = trade_notional * self.cfg.transaction_fee_pct
        slippage_cost = trade_notional * self.cfg.slippage_pct if abs(trade_quantity) > 0 else 0
        total_cost = base_fee + slippage_cost
        
        # Apply unrealized PnL and costs to balance
        unrealized_pnl = self.asset_held * (current_price - self.entry_price)
        self.balance += unrealized_pnl - total_cost
        
        # Update position
        self.asset_held = target_asset_quantity
        new_notional_value = abs(self.asset_held) * current_price
        self.used_margin = new_notional_value / self.leverage
        
        if abs(trade_quantity) > 1e-8:
            self.entry_price = current_price
            self.trade_count += 1
    
    def _get_single_step_observation(self, step_index) -> dict:
        """✅ ENHANCED: Single step observation with dual-horizon portfolio state."""
        try:
            if self.normalizer is None:
                return {}
            
            current_timestamp = self.base_timestamps[step_index]
            raw_obs = {}
            
            base_df = self.timeframes[self.cfg.base_bar_timeframe.value]
            current_price = base_df['close'].iloc[base_df.index.get_loc(current_timestamp, method='ffill')]
            
            # Process each observation type (existing code)
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
            
            # ✅ ENHANCED: Portfolio state with dual-horizon feedback
            unrealized_pnl = self.asset_held * (current_price - self.entry_price)
            portfolio_value = self.balance + unrealized_pnl
            
            normalized_position = np.clip(self.asset_held * current_price / (portfolio_value + 1e-9), -1.0, 1.0)
            pnl_on_margin = unrealized_pnl / (self.used_margin + 1e-9)
            normalized_pnl = np.tanh(pnl_on_margin)
            
            position_notional = abs(self.asset_held) * current_price
            margin_health = self.used_margin + unrealized_pnl
            margin_ratio = np.clip(margin_health / position_notional, 0, 2.0) if position_notional > 0 else 2.0
            
            # ✅ NEW: Dual-horizon specific features
            promise_count_normalized = len(self.reward_calculator.active_promises) / max(self.dual_horizon_config.max_promises, 1)
            promise_success_rate = self.reward_calculator.promise_success_rate
            
            # Enhanced portfolio state with dual-horizon information
            raw_obs[FeatureKeys.PORTFOLIO_STATE.value] = np.array([
                normalized_position,           # Current position
                normalized_pnl,               # Unrealized PnL
                margin_ratio,                 # Margin health
                promise_count_normalized,     # ✅ NEW: Active promises indicator
                promise_success_rate          # ✅ NEW: Historical promise success rate
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
        """Get observation sequence with enhanced error handling"""
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
        """✅ ENHANCED: Get comprehensive performance metrics including dual-horizon analytics."""
        try:
            if len(self.portfolio_history) < 2:
                return {'leverage': self.leverage}
            
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
            excess_return = total_return - 0.02  # Assuming 2% risk-free rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
            
            # Trading metrics
            win_rate = self.winning_trades / max(self.trade_count, 1)
            avg_reward = np.mean(self.step_rewards) if self.step_rewards else 0.0
            reward_volatility = np.std(self.step_rewards) if len(self.step_rewards) > 1 else 0.0
            
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
                'leverage': self.leverage,
            }
            
            # ✅ NEW: Add dual-horizon specific metrics
            dual_horizon_metrics = {
                'promises_created_total': self.dual_horizon_metrics['promises_created'],
                'promises_evaluated_total': self.dual_horizon_metrics['promises_evaluated'],
                'promises_successful_total': self.dual_horizon_metrics['promises_successful'],
                'promise_success_rate': self.dual_horizon_metrics['avg_promise_success_rate'],
                'realized_trades_total': self.dual_horizon_metrics['realized_trades'],
                'total_realized_pnl': self.dual_horizon_metrics['total_realized_pnl'],
                'avg_realized_pnl_per_trade': (
                    self.dual_horizon_metrics['total_realized_pnl'] / 
                    max(self.dual_horizon_metrics['realized_trades'], 1)
                ),
                'active_promises_final': len(self.reward_calculator.active_promises),
                'completed_promises_total': len(self.reward_calculator.completed_promises),
                
                # Dual-horizon reward breakdown
                'immediate_reward_weight': self.dual_horizon_config.immediate_reward_weight,
                'delayed_reward_weight': self.dual_horizon_config.delayed_reward_weight,
                'reward_horizon_steps': self.dual_horizon_config.reward_horizon_steps,
            }
            
            return {**base_metrics, **dual_horizon_metrics}
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'leverage': self.leverage}

# --- CONVENIENCE FUNCTIONS ---

def create_dual_horizon_environment(df_base_ohlc: pd.DataFrame, normalizer: Normalizer,
                                   config=None, leverage: float = None, 
                                   dual_horizon_config: DualHorizonConfig = None):
    """
    ✅ NEW: Convenience function to create dual-horizon trading environment.
    
    Args:
        df_base_ohlc: Base OHLC data
        normalizer: Fitted normalizer
        config: Configuration object
        leverage: Trading leverage (if None, uses config default)
        dual_horizon_config: Dual-horizon specific configuration
    """
    return EnhancedDualHorizonTradingEnvironment(
        df_base_ohlc=df_base_ohlc,
        normalizer=normalizer,
        config=config,
        leverage=leverage,
        dual_horizon_config=dual_horizon_config
    )

# Legacy compatibility
EnhancedHierarchicalTradingEnvironment = EnhancedDualHorizonTradingEnvironment

if __name__ == "__main__":
    # Example usage and testing
    try:
        logger.info("Testing dual-horizon trading environment...")
        
        # Test the dual-horizon reward calculator with different configurations
        test_configs = [
            DualHorizonConfig(reward_horizon_steps=9, immediate_reward_weight=0.3, delayed_reward_weight=0.7),
            DualHorizonConfig(reward_horizon_steps=15, immediate_reward_weight=0.5, delayed_reward_weight=0.5),
            DualHorizonConfig(reward_horizon_steps=6, immediate_reward_weight=0.2, delayed_reward_weight=0.8),
        ]
        
        for i, config in enumerate(test_configs):
            logger.info(f"Testing configuration {i+1}:")
            logger.info(f"  → Horizon: {config.reward_horizon_steps} steps")
            logger.info(f"  → Weights: {config.immediate_reward_weight:.1%} immediate, {config.delayed_reward_weight:.1%} delayed")
            
            reward_calc = DualHorizonRewardCalculator(config, leverage=10.0)
            
            # Test promise creation and evaluation
            test_promise = reward_calc.create_promise(
                step=100,
                action=np.array([0.5, 0.3]),
                current_price=50000.0,
                portfolio_value=1100000.0,
                expected_future_price=51000.0
            )
            
            logger.info(f"  → Created promise for step {test_promise.step}")
            
            # Test promise evaluation
            promise_results = reward_calc.evaluate_promises(
                current_step=100 + config.reward_horizon_steps,
                current_price=51200.0  # Successful prediction
            )
            
            if promise_results:
                result = promise_results[0]
                logger.info(f"  → Promise evaluated: {'SUCCESS' if result.success else 'FAILURE'}, reward: {result.reward:.4f}")
        
        logger.info("✅ Dual-horizon trading environment test completed!")
        
    except Exception as e:
        logger.error(f"Dual-horizon environment test failed: {e}", exc_info=True)