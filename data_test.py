import numpy as np
from datasourcing import *
from strategy_utils import *

def LS_equal(signals):
    signals = signals.sub(signals.median())
    signals[signals > 0] = signals[signals > 0].div(signals[signals > 0].sum())
    signals[signals < 0] = -(signals[signals < 0].div(signals[signals < 0].sum()))
    return signals

ticker_list = ["JNPR","AKAM","CRM","RL","VLO","META","TER","ADBE","FICO","MRNA"]
freq=2
delay=0

A = MarketData(tickers=ticker_list)
sample_data = A.get_historical(start_date="2022-05-10")
price = sample_data['Adj Close']
signals = (sample_data['High'] - sample_data['Low']).div(np.log(sample_data['Volume']))
signals = signals * 0.1 / (sample_data['Adj Close'].pct_change().rolling(20).std() * np.sqrt(252))

if __name__ == '__main__' :
    B = StockStrategy(signals=signals, price=price, freq=freq, delay=delay)
    B.regularisation()
    B.evaluate()
    B.save_performance_plot()



