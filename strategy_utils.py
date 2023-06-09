import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin.stock_info import tickers_sp500
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import os
import logging


class StockStrategy:
    def __init__(
        self, signals=pd.DataFrame(), price=pd.DataFrame(), delay=0, freq=1
    ):
        """signals:DataFrame, price:DataFrame"""
        assert len(signals) > 0 and len(price) > 1, "Missing data for Backtesting! "
        logging.info("Initialize the Strategy data...")
        if isinstance(signals, pd.Series):
            signals = signals.to_frame()
            signals.name = "Signals"
        if isinstance(price, pd.Series):
            price = price.to_frame()
            price.name = "Test Price"


        self.test_ret = price.div(price.shift(freq)).sub(1).shift(-freq-delay)
        self.price = price
        self.delay = delay
        self.freq = freq
        self.trade_days = [x for x in list(range(0, len(self.test_ret), freq))]
        self.test_ret = self.test_ret.iloc[self.trade_days, :].dropna(how="all")
        self.signals = signals.iloc[self.trade_days, :].dropna(how="all")

        if self.test_ret.shape[1] == 1:
            self.strategy_ret_all = (self.test_ret.mul(self.signals))
        else:
            self.strategy_ret_all = (self.test_ret.mul(self.signals)).sum(axis=1)

        assert len(self.test_ret) != 0, "Missing stock returns for Backtesting!"
        assert len(self.signals) != 0, "Missing signals for Backtesting!"

    # Define functions for scaling the signals
    def regularisation(self):
        def LS_equal(signals):
            signals = signals.sub(signals.median())
            signals[signals > 0] = signals[signals > 0].div(signals[signals > 0].sum())
            signals[signals < 0] = -(signals[signals < 0].div(signals[signals < 0].sum()))
            return signals

        if isinstance(self.signals, pd.DataFrame):
            self.signals = self.signals.apply(LS_equal, axis=1)

        if self.test_ret.shape[1] == 1:
            self.strategy_ret_all = (self.test_ret.mul(self.signals))
        else:
            self.strategy_ret_all = (self.test_ret.mul(self.signals)).sum(axis=1)

        return self.signals
    # Define functions for evaluating portfolio's performance
    def evaluate(self, riskfree=0):
        data = (1 + self.strategy_ret_all).cumprod()
        self.performance = data
        # compute max drawdown
        def Max_DD(data):
            answer = max(1 - data / data.cummax())
            return answer

        # compute annualized volatility
        def volatility(data):
            earning = data.pct_change().fillna(0)
            return earning.std() * np.sqrt(260 / self.freq)

        # compute return
        def all_return(data):
            return (data.iloc[-1] - data.iloc[0]) / data.iloc[0]

        # compute annualized return
        def annual_return(data):
            return np.power((1 + all_return(data)), (252 / self.freq / len(data))) - 1

        # compute sharpe ratio
        def Sharpe(data, risk_free=riskfree):
            return (annual_return(data) - risk_free) / volatility(data)

        # compute calmar ratio
        def Calmar(data, risk_free=riskfree):
            return (annual_return(data) - risk_free) / Max_DD(data)

        # compute return skew
        def Skew(data):
            return data.pct_change().skew()

        # compute return kurtosis
        def Kurt(data):
            return data.pct_change().kurtosis()


        def Win_rario(data):
            new = data.pct_change().dropna()
            return len(new[new > 0]) / (len(new[new < 0]) + len(new[new > 0]))

        # Get the date corresponding to max price
        def HWM_time(data):
            return data.idxmax().date()

        # Get max price

        def HWM(data):
            return data.max()

        # Get dates relative to Max Drawdown
        def DD_periods(data):
            DD = data.cummax() - data  # get all Drawdowns: diffs between cumulative max price and curent price
            end_mdd = DD.idxmax()  # get date of max Drawdown
            start_mdd = data[:end_mdd].idxmax()
            # data[:end_mdd]-prices from starting time to end_mdd Timestamp

            MDD = 1 - data[end_mdd] / data[start_mdd]
            # Maximum Drawdown as positive proportional loss from peak
            Peak_date = start_mdd.date()
            Trough_date = end_mdd.date()
            bool_P = data[end_mdd:] > data[start_mdd]
            # True/False current price > price of DD peak

            if (bool_P.idxmax().date() > bool_P.idxmin().date()):
                Rec_date = bool_P.idxmax().date()  # date of first True occurrence,
                # i.e. first time price goes over price of DD peak
                # is fully recovered from DD

                MDD_dur = (Rec_date - Peak_date).days  # MDD duration in days from peak to recovery date
            else:
                Rec_date = MDD_dur = 'Yet to recover'
                # IF no True found anywhere in bool_P series, then idxmax
            # As an error check, we check if idxmax = idxmin. In that case the rec date and mdd dur are not
            # known as the DD peak has NOT been recovered yet
            return [Peak_date, Trough_date, MDD_dur, Rec_date]

        # Store results
        if isinstance(data, pd.Series):
            data.name = "Strategy Performance"
            data = data.to_frame()

        results = pd.DataFrame(columns=data.columns, index=['All_ret', 'Annual_ret', 'Max_DD', 'Sharpe_ratio',
                                                            'Calmar_ratio', 'Skew', 'Kurt', 'HWM_time',
                                                            'HWM', 'Volatility', 'Win_Ratio', 'Peak_date',
                                                            'Trough_date', 'MDD_dur', 'Rec_date'])
        # Compute Strategy's result separately
        fct_data = data.iloc[:, 0].dropna()
        results.iloc[:, 0] = [all_return(fct_data), annual_return(fct_data), Max_DD(fct_data),
                              Sharpe(fct_data), Calmar(fct_data), Skew(fct_data), Kurt(fct_data),
                              HWM_time(fct_data), HWM(fct_data), volatility(fct_data),
                              Win_rario(fct_data)] + DD_periods(fct_data)
        logging.info(results)
        return results

    def save_performance_plot(self, name="Strategy Performance"):
        Figure = self.performance.plot()
        plt.grid(True)
        plt.show()
        plt.savefig(f"{name}.png")

