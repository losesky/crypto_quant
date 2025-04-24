"""
REST API服务模块，提供HTTP API接口
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
import json
import plotly.io as pio
from ...data.sources.binance_source import BinanceDataSource
from ...strategies.technical.macd_strategy import MACDStrategy
from ...strategies.ml_based.lstm_strategy import LSTMStrategy
from ...backtesting.engine.backtest_engine import BacktestEngine
from ...backtesting.visualization.performance_visualizer import PerformanceVisualizer
from ...utils.logger import logger
from ...config.settings import WEB_CONFIG


# 创建FastAPI应用
app = FastAPI(
    title="Crypto Quant API",
    description="比特币量化交易框架API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
data_source = BinanceDataSource()
results_cache = {}


# 请求模型
class BacktestRequest(BaseModel):
    """回测请求模型"""
    symbol: str = Field("BTC/USDT", description="交易对")
    interval: str = Field("1d", description="K线间隔")
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)")
    end_date: str = Field(None, description="结束日期 (YYYY-MM-DD)")
    strategy_type: str = Field("macd", description="策略类型 (macd, lstm)")
    strategy_params: dict = Field({}, description="策略参数")
    initial_capital: float = Field(10000.0, description="初始资金")
    commission: float = Field(0.001, description="手续费率")


class PredictionRequest(BaseModel):
    """预测请求模型"""
    symbol: str = Field("BTC/USDT", description="交易对")
    interval: str = Field("1d", description="K线间隔")
    days: int = Field(100, description="历史数据天数")
    prediction_days: int = Field(1, description="预测天数")
    model_params: dict = Field({}, description="模型参数")


# 响应模型
class BacktestResponse(BaseModel):
    """回测响应模型"""
    strategy_name: str
    performance: dict
    chart_url: str


class PredictionResponse(BaseModel):
    """预测响应模型"""
    symbol: str
    current_price: float
    predicted_prices: list
    signal: str
    chart_url: str


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("API服务启动")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("API服务关闭")


@app.get("/")
async def root():
    """API根路径"""
    return {"message": "欢迎使用比特币量化交易框架API"}


@app.get("/api/symbols")
async def get_symbols():
    """获取支持的交易对列表"""
    try:
        tickers = data_source.get_all_tickers()
        symbols = tickers['symbol'].tolist()
        return {"symbols": symbols}
    except Exception as e:
        logger.error(f"获取交易对列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    运行策略回测
    """
    try:
        # 获取数据
        logger.info(f"获取{request.symbol} {request.interval}数据: {request.start_date}至{request.end_date or '现在'}")
        df = data_source.get_historical_data(
            request.symbol, 
            request.interval, 
            start=request.start_date, 
            end=request.end_date
        )
        
        # 创建策略
        if request.strategy_type == "macd":
            fast = request.strategy_params.get("fast", 12)
            slow = request.strategy_params.get("slow", 26)
            signal = request.strategy_params.get("signal", 9)
            strategy = MACDStrategy(fast=fast, slow=slow, signal=signal)
        elif request.strategy_type == "lstm":
            sequence_length = request.strategy_params.get("sequence_length", 10)
            prediction_threshold = request.strategy_params.get("prediction_threshold", 0.01)
            strategy = LSTMStrategy(
                sequence_length=sequence_length, 
                prediction_threshold=prediction_threshold
            )
        else:
            raise HTTPException(status_code=400, detail=f"不支持的策略类型: {request.strategy_type}")
        
        # 创建回测引擎
        engine = BacktestEngine(
            df, 
            strategy, 
            initial_capital=request.initial_capital, 
            commission=request.commission
        )
        
        # 运行回测
        engine.run()
        
        # 获取回测结果
        performance = engine.summary()
        
        # 创建可视化
        visualizer = PerformanceVisualizer(engine.results, performance)
        dashboard = visualizer.create_interactive_dashboard()
        
        # 将Plotly图表转换为HTML
        chart_html = pio.to_html(dashboard, full_html=True)
        
        # 为图表生成唯一ID
        chart_id = f"{request.strategy_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        chart_path = f"static/charts/{chart_id}.html"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        
        # 保存HTML图表
        with open(chart_path, "w") as f:
            f.write(chart_html)
        
        # 缓存结果
        results_cache[chart_id] = {
            "strategy": strategy,
            "results": engine.results,
            "performance": performance
        }
        
        # 清理旧缓存
        background_tasks.add_task(cleanup_old_cache)
        
        return {
            "strategy_name": strategy.name,
            "performance": performance,
            "chart_url": f"/api/charts/{chart_id}"
        }
        
    except Exception as e:
        logger.error(f"回测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    预测价格
    """
    try:
        # 计算开始日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)
        start_str = start_date.strftime("%Y-%m-%d")
        
        # 获取数据
        logger.info(f"获取{request.symbol} {request.interval}数据用于预测: {start_str}至今")
        df = data_source.get_historical_data(
            request.symbol,
            request.interval,
            start=start_str
        )
        
        # 创建LSTM策略
        sequence_length = request.model_params.get("sequence_length", 10)
        prediction_threshold = request.model_params.get("prediction_threshold", 0.01)
        
        lstm_strategy = LSTMStrategy(
            sequence_length=sequence_length,
            prediction_threshold=prediction_threshold
        )
        
        # 训练模型
        lstm_strategy.train(df)
        
        # 预测未来价格
        predicted_prices = lstm_strategy.predict_next_day(df, n_steps=request.prediction_days)
        
        # 获取当前价格
        current_price = df['close'].iloc[-1]
        
        # 生成交易信号
        if predicted_prices:
            price_change = (predicted_prices[0] - current_price) / current_price
            
            if price_change > prediction_threshold:
                signal = "买入"
            elif price_change < -prediction_threshold:
                signal = "卖出"
            else:
                signal = "持有"
        else:
            signal = "无法预测"
        
        # 创建预测图表
        plt = lstm_strategy.predictor.plot_predictions(
            np.array(predicted_prices).reshape(-1, 1), 
            df['close'].iloc[-request.prediction_days:].values.reshape(-1, 1),
            f"{request.symbol} {request.prediction_days}天价格预测"
        )
        
        # 保存图表
        chart_id = f"prediction_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        chart_path = f"static/charts/{chart_id}.png"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        
        # 保存图表
        plt.savefig(chart_path)
        
        return {
            "symbol": request.symbol,
            "current_price": float(current_price),
            "predicted_prices": [float(p) for p in predicted_prices],
            "signal": signal,
            "chart_url": f"/api/images/{chart_id}"
        }
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/charts/{chart_id}")
async def get_chart(chart_id: str):
    """
    获取图表HTML
    """
    chart_path = f"static/charts/{chart_id}.html"
    if not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail="图表不存在")
        
    with open(chart_path, "r") as f:
        html_content = f.read()
        
    return HTMLResponse(content=html_content)


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """
    获取图像
    """
    from fastapi.responses import FileResponse
    
    image_path = f"static/charts/{image_id}.png"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="图像不存在")
        
    return FileResponse(image_path)


async def cleanup_old_cache():
    """
    清理旧缓存
    """
    # 获取所有缓存键
    keys = list(results_cache.keys())
    
    # 如果缓存数量超过20，删除最旧的项
    if len(keys) > 20:
        # 按照时间戳排序
        sorted_keys = sorted(keys, key=lambda k: k.split('_')[-1])
        
        # 删除最旧的10个项
        for key in sorted_keys[:10]:
            del results_cache[key]
            
            # 也删除相应的文件
            chart_path = f"static/charts/{key}.html"
            if os.path.exists(chart_path):
                os.remove(chart_path)


def start_api_server():
    """
    启动API服务器
    """
    # 确保静态文件目录存在
    os.makedirs("static/charts", exist_ok=True)
    
    # 从配置获取主机和端口
    host = WEB_CONFIG.get("host", "0.0.0.0")
    port = WEB_CONFIG.get("port", 8000)
    
    # 启动服务器
    uvicorn.run(
        "crypto_quant.api.rest.api_service:app", 
        host=host, 
        port=port,
        reload=WEB_CONFIG.get("debug", False)
    )


if __name__ == "__main__":
    start_api_server()