#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Regime Classifier Example

This script demonstrates how to use the new MarketRegimeClassifier
to identify different market states in Bitcoin price data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from crypto_quant package
from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.analysis.market_regime_classifier import MarketRegimeClassifier
from crypto_quant.indicators.technical_indicators import TechnicalIndicators
from crypto_quant.utils.logger import logger
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.utils.output_helper import get_image_path, get_report_path

# Get font helper instance
font_helper = get_font_helper()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """
    Main function to demonstrate the MarketRegimeClassifier
    """
    # Set parameters
    symbol = "BTC/USDT"
    interval = "1d"
    
    # Load 1.5 years of daily data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=545)  # ~1.5 years
    
    # Initialize data source
    logger.info("Initializing Binance data source...")
    data_source = BinanceDataSource()
    
    # Get historical data
    logger.info("Loading Bitcoin historical data...")
    df = data_source.get_historical_data(
        symbol=symbol,
        interval=interval,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d')
    )
    
    if df is None or len(df) < 100:
        logger.error("Failed to load sufficient data or data loading returned None")
        return
    
    logger.info(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")
    
    # Print data head
    print("\nData Sample:")
    print(df.head())
    
    # Initialize the market regime classifier
    classifier = MarketRegimeClassifier(
        adx_period=14,
        adx_threshold=25.0,
        rsi_period=14,
        rsi_thresholds=(30.0, 70.0),
        volatility_period=20,
        volatility_threshold=0.03,
        bb_period=20,
        bb_width_threshold=0.05,
        lookback_window=50,
        visualization_enabled=True
    )
    
    # Classify all data points
    logger.info("Classifying market regimes...")
    df_classified = classifier.classify_all(df)
    
    # Print the last 5 classified points
    print("\nLast 5 classified data points:")
    columns_to_show = ['close', 'adx', 'rsi', 'volatility', 'bb_width', 'market_state']
    print(df_classified[columns_to_show].tail())
    
    # Get statistics for each market state
    state_stats = classifier.get_state_statistics(df_classified)
    
    # Print state statistics
    print("\nMarket State Statistics:")
    for state, stats in state_stats.items():
        print(f"\n{state.upper()}:")
        print(f"  Count: {stats['count']} ({stats['percent']:.2f}% of data)")
        print(f"  Avg ADX: {stats['avg_adx']:.2f}")
        print(f"  Avg RSI: {stats['avg_rsi']:.2f}")
        print(f"  Avg Volatility: {stats['avg_volatility']:.4f}")
        print(f"  Avg BB Width: {stats['avg_bb_width']:.4f}")
        print(f"  Avg Return: {stats['avg_return']:.4f}%")
        print(f"  Median Return: {stats['median_return']:.4f}%")
        print(f"  Positive Return Ratio: {stats['pos_return_ratio']:.2f}%")
    
    # Get transition metrics
    transition_metrics = classifier.get_transition_metrics()
    print("\nMarket State Transition Metrics:")
    print(f"  Total Transitions: {transition_metrics['total_transitions']}")
    print(f"  Transitions per Day: {transition_metrics['transitions_per_day']:.4f}")
    print(f"  Average State Duration: {transition_metrics['average_state_duration']:.2f} periods")
    
    # Demonstrate strategy recommendations
    print("\nRecommended Strategy Parameters by Market State:")
    for state in ['strong_uptrend', 'strong_downtrend', 'volatile_range', 'tight_range']:
        strategy = classifier.get_recommended_strategy(state)
        print(f"\n{state.upper()} - {strategy['name']}: {strategy['description']}")
        print(f"  Parameters: {strategy['parameters']}")
        print(f"  Strategy Weights: {strategy['weights']}")
    
    # Create visualization
    logger.info("Creating visualization...")
    fig = classifier.visualize(df_classified, title="Bitcoin Market Regimes")
    
    # Apply Chinese font if available
    if font_helper.has_chinese_font:
        plt.suptitle("比特币市场状态分类", fontsize=16)
    else:
        plt.suptitle("Bitcoin Market Regime Classification", fontsize=16)
    
    # Save the visualization
    output_file = get_image_path('market_regimes_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    logger.info(f"Visualization saved to {output_file}")
    
    # Show the plot
    plt.show()
    
    # Example: Get current market state and corresponding strategy
    current_state = classifier.classify(df)
    current_strategy = classifier.get_recommended_strategy(current_state)
    
    print(f"\nCurrent Market State: {current_state}")
    print(f"Recommended Strategy: {current_strategy['name']}")
    print(f"Strategy Description: {current_strategy['description']}")
    print(f"Recommended Parameters: {current_strategy['parameters']}")
    print(f"Strategy Weights (MACD:LSTM): {current_strategy['weights']}")
    
    # Demonstrate different parameter settings
    print("\nTesting different classifier parameter settings...")
    
    # More sensitive to trends (lower ADX threshold)
    trend_sensitive_classifier = MarketRegimeClassifier(
        adx_threshold=20.0,
        visualization_enabled=False
    )
    
    # More sensitive to volatility
    volatility_sensitive_classifier = MarketRegimeClassifier(
        volatility_threshold=0.02,
        bb_width_threshold=0.04,
        visualization_enabled=False
    )
    
    # Classify with different settings
    df_trend_sensitive = trend_sensitive_classifier.classify_all(df)
    df_volatility_sensitive = volatility_sensitive_classifier.classify_all(df)
    
    # Compare last 10 days with different settings
    print("\nLast 10 days market states with different settings:")
    comparison_df = pd.DataFrame({
        'close': df.iloc[-10:]['close'],
        'standard': df_classified.iloc[-10:]['market_state'],
        'trend_sensitive': df_trend_sensitive.iloc[-10:]['market_state'],
        'volatility_sensitive': df_volatility_sensitive.iloc[-10:]['market_state']
    })
    print(comparison_df)
    
    # Compare state distribution
    print("\nState distribution with different settings:")
    
    def print_state_distribution(classifier_name, df):
        states = df['market_state'].value_counts(normalize=True) * 100
        print(f"\n{classifier_name}:")
        for state, percentage in states.items():
            print(f"  {state}: {percentage:.2f}%")
    
    print_state_distribution("Standard Settings", df_classified)
    print_state_distribution("Trend Sensitive", df_trend_sensitive)
    print_state_distribution("Volatility Sensitive", df_volatility_sensitive)
    
    logger.info("Market regime classification example completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main function: {str(e)}") 