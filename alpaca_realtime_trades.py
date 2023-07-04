import argparse
import asyncio
import websockets
import json
import matplotlib.pyplot as plt
import matplotlib.dates as md
from dateutil.parser import parse
from tzlocal import get_localzone
import os
import pandas as pd
import pandas_market_calendars as mcal


class AlpacaStream:
    """Class for streaming real-time trades from Alpaca."""

    def __init__(self, symbol):
        """Initialize the AlpacaStream with a symbol."""
        self.API_KEY = os.getenv("APCA_API_KEY_ID")
        self.API_SECRET = os.getenv("APCA_API_SECRET_KEY")
        self.URL = "wss://stream.data.alpaca.markets/v2/sip"
        self.symbol = symbol.upper()
        self.local_tz = get_localzone()

        self.fig, self.axs = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(14, 8),
        )
        self.xdata, self.ydata, self.vdata = [], [], []
        (self.ln_price,) = self.axs[0].plot([], [], "r-", label=self.symbol)
        self.axs[0].set_ylabel("Trade Price")
        self.axs[0].grid(
            True, which="both", color="lightgrey", linestyle="--"
        )
        self.axs[0].legend(loc="best")
        self.axs[1].bar(
            self.xdata,
            self.vdata,
            width=0.0001,
            align="center",
            color="b",
            label="Volume",
        )
        self.axs[1].set_xlabel("Time")
        self.axs[1].set_ylabel("Trade Volume")
        self.axs[1].grid(
            True, which="both", color="lightgrey", linestyle="--"
        )
        self.axs[1].legend(loc="best")
        plt.tight_layout()
        plt.ion()

    def rfc3339_to_local_datetime(self, rfc3339_str):
        """Convert a RFC 3339 timestamp to local time."""
        dt = parse(rfc3339_str)
        dt = dt.astimezone(self.local_tz)
        dt = dt.replace(tzinfo=None)
        return dt

    def update_plot(self, new_x, new_y, new_v):
        """Update the plot with new data."""
        self.xdata.append(new_x)
        self.ydata.append(new_y)
        self.vdata.append(new_v)
        self.ln_price.set_data(self.xdata, self.ydata)
        self.axs[0].relim()
        self.axs[0].autoscale_view()
        self.axs[0].xaxis_date()
        self.axs[0].xaxis.set_major_formatter(md.DateFormatter("%H:%M:%S"))
        self.axs[1].clear()
        self.axs[1].bar(
            self.xdata,
            self.vdata,
            width=0.0001,
            align="center",
            color="b",
            label="Volume",
        )
        self.axs[1].legend(loc="best")
        self.axs[1].xaxis_date()
        self.axs[1].xaxis.set_major_formatter(md.DateFormatter("%H:%M:%S"))
        self.axs[1].grid(
            True, which="both", color="lightgrey", linestyle="--"
        )
        self.fig.suptitle(
            f"Real-time {self.symbol} Trades - Price: {new_y} Volume: {new_v}",
            fontsize=12,
        )
        plt.draw()
        plt.pause(0.01)

    async def handle_trades(self, websocket):
        """Handle incoming trade messages."""
        await websocket.send(
            json.dumps(
                {
                    "action": "auth",
                    "key": self.API_KEY,
                    "secret": self.API_SECRET,
                }
            )
        )
        await websocket.send(
            json.dumps({"action": "subscribe", "trades": [self.symbol]})
        )
        while True:
            message = await websocket.recv()
            msg_json = json.loads(message)
            for msg in msg_json:
                if msg["T"] == "t":
                    print(msg)
                    new_x = md.date2num(
                        self.rfc3339_to_local_datetime(msg["t"])
                    )
                    new_y = msg["p"]
                    new_v = msg["s"]
                    self.update_plot(new_x, new_y, new_v)

    async def connect(self):
        """Open the websocket connection and handle trades."""
        async with websockets.connect(self.URL) as websocket:
            await self.handle_trades(websocket)

    def run(self):
        # Check if the market is open
        nyse = mcal.get_calendar("NYSE")
        market_schedule = nyse.schedule(
            start_date=pd.Timestamp.today(), end_date=pd.Timestamp.today()
        )
        market_open_today = not market_schedule.empty

        if market_open_today:
            plt.show()
            asyncio.run(self.connect())
        else:
            print("The market is closed today. Exiting the program.")
            exit()

        """Start the Alpaca stream."""
        plt.show()
        asyncio.run(self.connect())


# Parse the command line argument for a stock symbol
parser = argparse.ArgumentParser(
    description="Stream trades for a specific symbol."
)
parser.add_argument(
    "--symbol",
    "-s",
    type=str,
    required=True,
    help="The stock symbol to stream trades for.",
)
args = parser.parse_args()

# Create an AlpacaStream and start it
stream = AlpacaStream(args.symbol)
stream.run()
