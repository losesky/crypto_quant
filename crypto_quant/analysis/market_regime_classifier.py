"""
Market Regime Classifier

This module provides advanced market state classification functionality,
using multiple technical indicators to identify different market regimes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

class MarketRegimeClassifier:
    """
    Advanced market regime classifier using multiple indicators to identify market states.
    
    The classifier categorizes market conditions into four main states:
    1. Strong Uptrend: Strong directional movement upward
    2. Strong Downtrend: Strong directional movement downward
    3. High Volatility Range: Wide price swings without clear direction
    4. Low Volatility Range: Narrow price range with minimal movement
    
    Parameters:
    -----------
    adx_period: int
        Period for ADX calculation (trend strength)
    adx_threshold: float
        Threshold for considering a market trending (typically 20-25)
    rsi_period: int
        Period for RSI calculation
    rsi_thresholds: Tuple[float, float]
        Lower and upper thresholds for RSI (typically 30, 70)
    volatility_period: int
        Period for volatility calculation
    volatility_threshold: float
        Threshold for high vs low volatility
    bb_period: int
        Period for Bollinger Bands calculation
    bb_width_threshold: float
        Threshold for Bollinger Bands width to identify ranging markets
    lookback_window: int
        Window size for considering historical data points
    visualization_enabled: bool
        Whether to enable visualization features
    """
    
    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_period: int = 14,
        rsi_thresholds: Tuple[float, float] = (30.0, 70.0),
        volatility_period: int = 20,
        volatility_threshold: float = 0.03,
        bb_period: int = 20,
        bb_width_threshold: float = 0.05,
        lookback_window: int = 50,
        visualization_enabled: bool = True
    ):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.rsi_thresholds = rsi_thresholds
        self.volatility_period = volatility_period
        self.volatility_threshold = volatility_threshold
        self.bb_period = bb_period
        self.bb_width_threshold = bb_width_threshold
        self.lookback_window = lookback_window
        self.visualization_enabled = visualization_enabled
        
        # Track historical market states
        self.market_state_history = []
        self.state_transitions = 0
        self.last_state = None
        
        logger.info(
            f"Market Regime Classifier initialized with: ADX({adx_period}, {adx_threshold}), "
            f"RSI({rsi_period}, {rsi_thresholds}), Volatility({volatility_period}, {volatility_threshold}), "
            f"BB({bb_period}, {bb_width_threshold})"
        )
    
    def classify(self, df: pd.DataFrame, index: Optional[int] = None) -> str:
        """
        Classify the market regime at a specific index or the latest data point.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Market data containing OHLCV information
        index: Optional[int]
            Specific index to classify. If None, classifies the last available data point.
            
        Returns:
        --------
        str: Market regime classification
            One of: 'strong_uptrend', 'strong_downtrend', 'volatile_range', 'tight_range'
        """
        # Calculate all necessary indicators
        df_with_indicators = self._calculate_indicators(df)
        
        # Get index to classify
        if index is None or index >= len(df_with_indicators):
            index = len(df_with_indicators) - 1
        
        # Extract feature values
        try:
            adx_value = df_with_indicators['adx'].iloc[index]
            rsi_value = df_with_indicators['rsi'].iloc[index]
            plus_di = df_with_indicators['plus_di'].iloc[index]
            minus_di = df_with_indicators['minus_di'].iloc[index]
            volatility = df_with_indicators['volatility'].iloc[index]
            bb_width = df_with_indicators['bb_width'].iloc[index]
            
            # Determine market regime
            is_trending = adx_value > self.adx_threshold
            is_volatile = volatility > self.volatility_threshold or bb_width > self.bb_width_threshold
            
            # Classify based on indicators
            if is_trending:
                # Trending market
                if plus_di > minus_di:
                    market_state = 'strong_uptrend'
                else:
                    market_state = 'strong_downtrend'
            else:
                # Ranging market
                if is_volatile:
                    market_state = 'volatile_range'
                else:
                    market_state = 'tight_range'
            
            # Track transitions
            if self.last_state is not None and self.last_state != market_state:
                self.state_transitions += 1
            
            # Update history
            self.last_state = market_state
            self.market_state_history.append((df_with_indicators.index[index], market_state))
            
            logger.debug(
                f"Market state at {df_with_indicators.index[index]}: {market_state} "
                f"[ADX={adx_value:.2f}, RSI={rsi_value:.2f}, Volatility={volatility:.4f}, BB Width={bb_width:.4f}]"
            )
            
            return market_state
            
        except Exception as e:
            logger.error(f"Error classifying market state: {str(e)}")
            return "unknown"
    
    def classify_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify the market regime for all data points.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Market data containing OHLCV information
            
        Returns:
        --------
        pd.DataFrame: DataFrame with added columns for market state and indicators
        """
        # Calculate all necessary indicators
        df_with_indicators = self._calculate_indicators(df)
        
        # Reset history
        self.market_state_history = []
        self.state_transitions = 0
        self.last_state = None
        
        # Initialize market state column
        df_with_indicators['market_state'] = 'unknown'
        
        # Classify each point
        for i in range(self.lookback_window, len(df_with_indicators)):
            df_with_indicators.loc[df_with_indicators.index[i], 'market_state'] = self.classify(df_with_indicators, i)
        
        # Map state to numeric values for easier analysis
        state_map = {
            'strong_uptrend': 3,
            'volatile_range': 2,
            'tight_range': 1,
            'strong_downtrend': 0,
            'unknown': -1
        }
        df_with_indicators['market_state_numeric'] = df_with_indicators['market_state'].map(state_map)
        
        logger.info(f"Classified {len(df_with_indicators)} data points with {self.state_transitions} state transitions")
        
        return df_with_indicators
    
    def get_state_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each market state.
        
        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame with market_state column
            
        Returns:
        --------
        Dict: Statistics for each market state
        """
        if 'market_state' not in df.columns:
            df = self.classify_all(df)
        
        result = {}
        
        # Get unique states
        states = df['market_state'].unique()
        
        for state in states:
            if state == 'unknown':
                continue
                
            # Get data for this state
            state_df = df[df['market_state'] == state]
            
            # Skip if not enough data
            if len(state_df) < 5:
                continue
            
            # Calculate statistics
            state_stats = {
                'count': len(state_df),
                'percent': len(state_df) / len(df) * 100,
                'avg_volatility': state_df['volatility'].mean(),
                'avg_adx': state_df['adx'].mean(),
                'avg_rsi': state_df['rsi'].mean(),
                'avg_bb_width': state_df['bb_width'].mean(),
                'avg_return': state_df['close'].pct_change().mean() * 100,
                'median_return': state_df['close'].pct_change().median() * 100,
                'pos_return_ratio': (state_df['close'].pct_change() > 0).mean() * 100,
            }
            
            result[state] = state_stats
        
        return result
    
    def visualize(self, df: pd.DataFrame, title: str = "Market Regime Classification") -> plt.Figure:
        """
        Visualize the market states on price chart.
        
        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame with market data and state classifications
        title: str
            Title for the chart
            
        Returns:
        --------
        plt.Figure: Matplotlib figure with the visualization
        """
        if not self.visualization_enabled:
            logger.warning("Visualization is disabled")
            return None
            
        # Ensure we have market state data
        if 'market_state' not in df.columns:
            df = self.classify_all(df)
        
        # Create figure
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot price
        axs[0].plot(df.index, df['close'], label='Price', color='black', alpha=0.7)
        
        # Color background based on market state
        for state in ['strong_uptrend', 'strong_downtrend', 'volatile_range', 'tight_range']:
            mask = df['market_state'] == state
            if not any(mask):
                continue
                
            # Choose color based on state
            if state == 'strong_uptrend':
                color = 'green'
                alpha = 0.3
            elif state == 'strong_downtrend':
                color = 'red'
                alpha = 0.3
            elif state == 'volatile_range':
                color = 'orange'
                alpha = 0.2
            else:  # tight_range
                color = 'blue'
                alpha = 0.1
            
            # Plot background color
            axs[0].fill_between(df.index, df['close'].min(), df['close'].max(), 
                              where=mask, color=color, alpha=alpha, label=state)
        
        axs[0].set_title(f"{title} - Market Regimes")
        axs[0].set_ylabel("Price")
        axs[0].legend(loc='upper left')
        axs[0].grid(True, alpha=0.3)
        
        # Plot ADX
        axs[1].plot(df.index, df['adx'], label='ADX', color='purple')
        axs[1].plot(df.index, df['plus_di'], label='+DI', color='green')
        axs[1].plot(df.index, df['minus_di'], label='-DI', color='red')
        axs[1].axhline(self.adx_threshold, linestyle='--', color='gray')
        axs[1].set_ylabel("ADX")
        axs[1].legend(loc='upper left')
        axs[1].grid(True, alpha=0.3)
        
        # Plot RSI and Volatility
        ax2 = axs[2]
        ax2.plot(df.index, df['rsi'], label='RSI', color='blue')
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.axhline(self.rsi_thresholds[0], linestyle='--', color='green')
        ax2.axhline(self.rsi_thresholds[1], linestyle='--', color='red')
        
        # Add volatility on secondary axis
        ax3 = ax2.twinx()
        ax3.plot(df.index, df['volatility'] * 100, label='Volatility', color='orange')
        ax3.axhline(self.volatility_threshold * 100, linestyle='--', color='gray')
        ax3.set_ylabel("Volatility %")
        
        # Create combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all necessary technical indicators for classification.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Market data containing OHLCV information
            
        Returns:
        --------
        pd.DataFrame: Original dataframe with added indicator columns
        """
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Import from TechnicalIndicators class if available
        try:
            from ..indicators.technical_indicators import TechnicalIndicators
            
            # ADX calculation
            df_copy['adx'] = TechnicalIndicators.calculate_adx(df_copy, period=self.adx_period)
            
            # Get +DI and -DI for trend direction
            high = df_copy['high'].values
            low = df_copy['low'].values
            close = df_copy['close'].values
            
            # Calculate DirectionalMovement +DM and -DM
            plus_dm = np.zeros(len(df_copy))
            minus_dm = np.zeros(len(df_copy))
            
            for i in range(1, len(df_copy)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                elif down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
            
            # Smooth DM values
            period = self.adx_period
            plus_di = np.zeros(len(df_copy))
            minus_di = np.zeros(len(df_copy))
            
            # Calculate true range first
            tr = np.zeros(len(df_copy))
            for i in range(1, len(df_copy)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
            
            # Calculate initial values
            plus_dm_sum = np.sum(plus_dm[1:period+1])
            minus_dm_sum = np.sum(minus_dm[1:period+1])
            tr_sum = np.sum(tr[1:period+1])
            
            plus_di[period] = 100 * plus_dm_sum / tr_sum if tr_sum != 0 else 0
            minus_di[period] = 100 * minus_dm_sum / tr_sum if tr_sum != 0 else 0
            
            # Calculate subsequent values using Wilder's smoothing
            for i in range(period+1, len(df_copy)):
                plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
                minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
                tr_sum = tr_sum - (tr_sum / period) + tr[i]
                
                plus_di[i] = 100 * plus_dm_sum / tr_sum if tr_sum != 0 else 0
                minus_di[i] = 100 * minus_dm_sum / tr_sum if tr_sum != 0 else 0
            
            df_copy['plus_di'] = plus_di
            df_copy['minus_di'] = minus_di
            
            # RSI calculation
            df_copy['rsi'] = TechnicalIndicators.calculate_rsi(df_copy, period=self.rsi_period)
            
            # Volatility
            volatility = TechnicalIndicators.calculate_volatility(df_copy, period=self.volatility_period)
            df_copy['volatility'] = df_copy['close'].pct_change().rolling(window=self.volatility_period).std()
            
            # Bollinger Bands
            upper, middle, lower = TechnicalIndicators.calculate_bb_bands(df_copy, period=self.bb_period)
            df_copy['bb_upper'] = upper
            df_copy['bb_middle'] = middle
            df_copy['bb_lower'] = lower
            df_copy['bb_width'] = (upper - lower) / middle
            
        except ImportError:
            logger.warning(
                "TechnicalIndicators class not available. "
                "Falling back to internal implementation."
            )
            self._calculate_indicators_internal(df_copy)
        
        # Clean up NaN values
        df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
        
        return df_copy
    
    def _calculate_indicators_internal(self, df: pd.DataFrame) -> None:
        """
        Internal implementation of indicator calculations.
        This is used as a fallback if TechnicalIndicators is not available.
        
        Parameters:
        -----------
        df: pd.DataFrame
            Market data containing OHLCV information
        """
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate volatility
        df['volatility'] = df['close'].pct_change().rolling(window=self.volatility_period).std()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (std * 2)
        df['bb_lower'] = df['bb_middle'] - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate ADX
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # We'll use a simple ADX calculation here
        # In reality, ADX calculation is more complex and requires several steps
        
        # Calculate true range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.zeros(len(df))
        tr[1:] = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate +DM and -DM
        plus_dm = np.zeros(len(df))
        minus_dm = np.zeros(len(df))
        
        for i in range(1, len(df)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            elif down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Smooth with EMA
        smoothing = 2.0 / (self.adx_period + 1)
        
        plus_di = pd.Series(plus_dm).ewm(alpha=smoothing, adjust=False).mean() / pd.Series(tr).ewm(alpha=smoothing, adjust=False).mean() * 100
        minus_di = pd.Series(minus_dm).ewm(alpha=smoothing, adjust=False).mean() / pd.Series(tr).ewm(alpha=smoothing, adjust=False).mean() * 100
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).ewm(alpha=smoothing, adjust=False).mean()
        
        df['plus_di'] = plus_di.values
        df['minus_di'] = minus_di.values
        df['adx'] = adx.values
    
    def get_recommended_strategy(self, market_state: str) -> Dict[str, Union[str, Dict]]:
        """
        Get recommended trading strategy based on market state.
        
        Parameters:
        -----------
        market_state: str
            The classified market state
            
        Returns:
        --------
        Dict: Recommended strategy and parameters
        """
        strategies = {
            'strong_uptrend': {
                'name': 'trend_following',
                'description': 'Follow the strong uptrend with momentum strategy',
                'parameters': {
                    'entry_threshold': 0.01,
                    'stop_loss': 0.05,
                    'take_profit': 0.15,
                    'position_size': 1.0,
                    'trailing_stop': True
                },
                'weights': {
                    'macd': 0.7,
                    'lstm': 0.3
                }
            },
            'strong_downtrend': {
                'name': 'trend_following_short',
                'description': 'Follow the strong downtrend with short positions',
                'parameters': {
                    'entry_threshold': -0.01,
                    'stop_loss': 0.04,
                    'take_profit': 0.12,
                    'position_size': 0.8,
                    'trailing_stop': True
                },
                'weights': {
                    'macd': 0.7,
                    'lstm': 0.3
                }
            },
            'volatile_range': {
                'name': 'mean_reversion',
                'description': 'Trade mean reversion in volatile ranging markets',
                'parameters': {
                    'lookback': 5,
                    'entry_threshold': 2.0,
                    'stop_loss': 0.03,
                    'take_profit': 0.06,
                    'position_size': 0.6,
                    'trailing_stop': False
                },
                'weights': {
                    'macd': 0.2,
                    'lstm': 0.8
                }
            },
            'tight_range': {
                'name': 'breakout',
                'description': 'Look for breakouts from tight ranges',
                'parameters': {
                    'channel_period': 20,
                    'entry_threshold': 0.02,
                    'stop_loss': 0.02,
                    'take_profit': 0.06,
                    'position_size': 0.5,
                    'trailing_stop': False
                },
                'weights': {
                    'macd': 0.5,
                    'lstm': 0.5
                }
            },
            'unknown': {
                'name': 'conservative',
                'description': 'Conservative approach when market state is uncertain',
                'parameters': {
                    'entry_threshold': 0.03,
                    'stop_loss': 0.02,
                    'take_profit': 0.04,
                    'position_size': 0.3,
                    'trailing_stop': False
                },
                'weights': {
                    'macd': 0.5,
                    'lstm': 0.5
                }
            }
        }
        
        return strategies.get(market_state, strategies['unknown'])
    
    def get_transition_metrics(self) -> Dict[str, int]:
        """
        Get metrics about market state transitions
        
        Returns:
        --------
        Dict: Transition metrics
        """
        return {
            'total_transitions': self.state_transitions,
            'transitions_per_day': self.state_transitions / (len(self.market_state_history) / 24) if self.market_state_history else 0,
            'average_state_duration': len(self.market_state_history) / (self.state_transitions + 1) if self.market_state_history else 0
        } 