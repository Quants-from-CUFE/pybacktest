import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin.stock_info import tickers_sp500
from datetime import datetime, timedelta
import pickle
import os
import logging
import abc
import requests
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class Data:
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_historical(self):
        pass

    @abc.abstractmethod
    def get_intraday(self):
        pass

    @abc.abstractmethod
    def cache_data(self):
        pass

    @abc.abstractmethod
    def restore_data(self):
        pass


class LocalData(Data):
    def __init__(self, path):
        self.path = path

    def restore_data(self, name):
        self.df = pd.read_pickle(f"{self.path}/{name}")
        return self.df

    def read_csv(self, name):
        self.df = pd.read_csv(f"{self.path}/{name}")
        return self.df


class YahooFinanceData(Data):
    def __init__(self, tickers=tickers_sp500(), path=os.getcwd()):
        self.tickers = tickers
        self.path = path

    def reset_tickers(self, tickers):
        self.tickers = tickers

    def reset_path(self, path):
        self.path

    def get_historical(self, start_date, end_date):
        self.df = yf.download(" ".join(self.tickers), start=start_date, end=end_date)
        return self.df

    def cache_data(self, name):
        filename = f"cache_{name}_yahoo_{str(datetime.now().date())}.pkl"
        logging.info("Saving cache to {filename}")
        with open(r"{self.path}/{filename}") as f:
            pickle.dump(self.df, f)


class alphaVantage(Data):
    """Stock data"""

    def __init__(self, key, tickers, path=os.getcwd()):
        self.tickers = tickers
        self.path = path
        self.apikey = key

    def get_historical(self, start_date, end_date, sleep=60):
        dfs = []
        for ticker in tqdm(self.tickers):
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outsize=full&apikey={self.apikey}"
            try:
                r = requests.get(url)
                data = r.json()
            except KeyError:
                if "Note" in data.keys():
                    time.sleep(sleep)
                    r = requests.get(url)
                    data = r.json()
                else:
                    break
            df = pd.DataFrame(data["Time Series (Daily)"]).T
            df["symbol"] = ticker
            dfs.append(df)
            # time.sleep(sleep)
        dfs = pd.concat(dfs, axis=0)
        dfs.index = pd.to_datetime(dfs.index)
        dfs = dfs.reset_index().rename(
            columns={
                "index": "date",
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adj Close",
                "6. volume": "Volume",
            }
        )

        self.df = dfs[
            ["date", "symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        ]
        self.df = self.df.set_index(["date", "symbol"]).unstack()
        self.df = self.df[
            (self.df.index >= datetime.strptime(start_date, "%Y-%m-%d"))
            & (self.df.index <= datetime.strptime(end_date, "%Y-%m-%d"))
        ]
        return self.df

    def cache_data(self, name):
        filename = f"cache_{name}_alphavantage_{str(datetime.now().date())}.pkl"
        logging.info("Saving cache to {filename}")
        with open(r"{self.path}/{filename}") as f:
            pickle.dump(self.df, f)


class StockData(Data):
    def __init__(self, key, tickers, path=os.getcwd()):
        ## TO DO
        self.tickers = tickers
        self.path = path
        self.apikey = key

    def get_historical(self, start_date, end_date):
        """maximum 3 tickers for free plan"""
        tickers = ",".join(self.tickers)
        url = f"https://api.stockdata.org/v1/data/eod?symbols={tickers}&date_from={str(start_date)}&date_to={str(end_date)}&api_token={self.apikey}"
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data["data"])
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Adj Close",
                "volume": "Volume",
            }
        )
        df.date = pd.to_datetime(df.date)
        df.set_index("date", inplace=True)
        self.df = df
        return self.df

    def cache_data(self, name):
        name = f"cache_{name}_stockdata_{str(datetime.now().date())}.pkl"
        logging.info("Saving cache to {filename}")
        with open(r"{self.path}/{filename}") as f:
            pickle.dump(self.df, f)


class MarketData:
    def __init__(
        self,
        apikey="",
        tickers=tickers_sp500(),
        source="yahoo_finance",
        path=os.getcwd(),
    ):
        """tickers:list, source:list"""
        assert source in [
            "yahoo_finance",
            "local",
            "alphavantage",
            "stockdata",
        ], "Source not in the list!"
        logging.info("Initialize the market data...")
        self.source = source
        if source == "yahoo_finance":
            self.obj = YahooFinanceData(tickers, path)
        elif source == "local":
            self.obj = LocalData(path)
        elif source == "alphavantage":
            self.obj = alphaVantage(apikey, tickers, path)
        elif source == "stockdata":
            self.obj = StockData(apikey, tickers, path)

    def get_historical(
        self, start_date, end_date=str((datetime.today() - timedelta(days=1)).date())
    ):
        logging.info(
            f"Get historical data from {self.source}, date from {start_date} to {end_date}..."
        )
        self.df = self.obj.get_historical(start_date, end_date)
        self.fields_names = list(set(x[0] for x in self.df.columns))
        return self.df
