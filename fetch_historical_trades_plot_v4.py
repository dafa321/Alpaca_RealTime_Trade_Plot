import argparse
import os
import pandas as pd
import requests
from datetime import date
from dateutil.parser import parse
from tzlocal import get_localzone
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np


class HistoricalTradePlotter:
    """Fetches historical trade data from the Alpaca API and plots the data."""

    BASE_URL = "https://data.alpaca.markets/v2/stocks"

    def __init__(self, symbol, limit=10000):
        self.symbol = symbol
        self.limit = limit
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("APCA_API_SECRET_KEY")
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }
        self.local_tz = get_localzone()

    def convert_timestamp_to_local_datetime(self, rfc3339_str):
        """Convert a RFC 3339 timestamp to local time."""
        dt = parse(rfc3339_str)
        dt = dt.astimezone(self.local_tz)
        dt = dt.replace(tzinfo=None)
        return dt

    def plot_data(self, df):
        """Plot the fetched data."""
        fig, axs = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(14, 8),
        )

        axs[0].plot(df["t"], df["p"], "r-", label=self.symbol)
        axs[0].set_ylabel("Trade Price")
        axs[0].grid(True, which="both", color="lightgrey", linestyle="--")
        axs[0].legend(loc="best")
        axs[0].xaxis_date()
        axs[0].xaxis.set_major_formatter(md.DateFormatter("%H:%M:%S"))
        axs[0].set_title(
            f"{self.symbol} Historical Trades for the Day - Current Trade Price: {df['p'].iloc[-1]}"
        )

        axs[1].bar(
            df["t"],
            np.where(df["s"] > 0, np.log(df["s"]), 0),
            width=0.0001,
            align="center",
            color="b",
            label="Volume (log scale)",
        )
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Trade Volume (log scale)")
        axs[1].grid(True, which="both", color="lightgrey", linestyle="--")
        axs[1].legend(loc="best")
        axs[1].xaxis_date()
        axs[1].xaxis.set_major_formatter(md.DateFormatter("%H:%M:%S"))

        plt.tight_layout()
        plt.show()


    def fetch_and_plot_trades(self):
        """Fetches historical trades data and plots the data."""
        today = date.today().isoformat()
        url = f"{self.BASE_URL}/{self.symbol}/trades?start={today}&limit={self.limit}"
        trades = []
        next_page_token = None

        while True:
            if next_page_token:
                response = requests.get(
                    url + f"&page_token={next_page_token}", headers=self.headers
                )
            else:
                response = requests.get(url, headers=self.headers)

            response.raise_for_status()
            data = response.json()

            for trade in data["trades"]:
                trades.append({"t": trade["t"], "p": trade["p"], "s": trade["s"]})

            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

        df = pd.DataFrame(trades)

        df = df[df["s"] > 0]

        df["t"] = df["t"].apply(self.convert_timestamp_to_local_datetime)
        df["t"] = df["t"].apply(md.date2num)

        # remove any duplicate consecutive prices
        df = df.loc[df["p"].shift() != df["p"]]

        self.plot_data(df)
        return df


# Parse command line argument for a stock symbol
parser = argparse.ArgumentParser(
    description="Fetch and plot trade data for a specific symbol."
)
parser.add_argument(
    "--symbol",
    "-s",
    type=str,
    required=True,
    help="The stock symbol to fetch trades for.",
)
args = parser.parse_args()

# Create an HistoricalTradePlotter and fetch trades
plotter = HistoricalTradePlotter(args.symbol.upper())
plotter.fetch_and_plot_trades()
