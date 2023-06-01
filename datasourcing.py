import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin.stock_info import tickers_sp500
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)


class MarketData:
    def __init__(
        self, tickers=tickers_sp500(), source="yahoo_finance", path=os.getcwd()
    ):
        """tickers:list, source:list ('yahoo_finance','local')"""
        assert source in ["yahoo_finance", "local"], "Source not in the list!"
        logging.info("Initialize the market data...")
        self.tickers = tickers
        self.source = source
        self.path = path

    def get_historical(
        self,
        start_date=str(datetime.now().date() - timedelta(2)),
        end_date=str(datetime.now().date() - timedelta(1)),
        filename="",
        filetype="csv",
    ):
        logging.info(
            f"Get historical data from {self.source}, date from {start_date} to {end_date}..."
        )
        if self.source == "yahoo_finance":
            self.df = self.get_yahoo_historical(
                start_date=start_date, end_date=end_date
            )
            self.datalist = list(set(x[0] for x in self.df.columns))

        if self.source == "local":
            self.df = self.get_local(filename=filename, filetype=filetype)
        return self.df

    def get_yahoo_historical(self, start_date, end_date):
        """date in %Y-%m-%d format"""
        df = yf.download(" ".join(self.tickers), start=start_date, end=end_date)
        return df

    def get_local(self, filename, filetype="csv"):
        if filetype == "csv":
            return pd.read_csv(filename)
        elif filetype == "pickle":
            return pd.read_pickle(filename)

    def cached_data(self, filename=f"cache_{str(datetime.now().date())}.pkl"):
        logging.info("Saving cache to {filename}")
        self.df.to_csv(f"{self.path}/{filename}")
