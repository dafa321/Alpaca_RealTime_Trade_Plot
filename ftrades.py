#!/usr/bin/python3

import argparse
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from alpaca_trade_api.rest import REST
from peakdetect import peakdetect
from scipy import signal
from matplotlib.style import use
use('dark_background')

# Get the Alpaca keys from system environment variables
API_KEY_ID = os.environ["APCA_API_KEY_ID"]
SECRET_KEY_ID = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)

pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", None, "display.max_columns", None)

look_ahead = 1

def tolgi_filtering(
    data: np.ndarray, num_samples: int, filter_window: int
) -> Tuple[np.ndarray, float, float]:
    """
    Apply a Hanning window filter to the given data and remove the trend line.

    Parameters:
    data (ndarray): 1D array of data points.
    num_samples (int): Number of samples in the data.
    filter_window (int): Window size of the Hanning window filter.

    Returns:
    tuple: Tuple containing the filtered data, gradient, and intercept of the trend line.
    """
    # Compute the gradient and intercept of the trend line
    gradient = (data[-1] - data[0]) / (num_samples - 1)
    intercept = data[0]

    # Generate an array of values for the trend line
    x = range(num_samples)
    trend = gradient * x + intercept

    # Remove the trend from the data
    remove_trend = data - trend

    # Generate the Hann window filter manually
    generated_filter = 0.5 * (
        1 - np.cos(2 * np.pi * np.arange(filter_window) / (filter_window - 1))
    )

    # Compute the convolution using the convolve function
    filtered_result = signal.convolve(remove_trend, generated_filter, mode="same")

    # Normalize the result
    filtered_result /= np.sum(generated_filter)

    # Add the trend back to the filtered data
    data_filter = filtered_result + trend

    # Return the filtered data, gradient, and intercept of the trend line
    return data_filter, gradient, intercept


def compute_avo_attributes(data):
    """
    Compute the gradient and intercept of the least squares fit line for the given data.

    Parameters:
    data (ndarray): 1D array of data points.

    Returns:
    tuple: Tuple containing the gradient and intercept of the least squares fit line.
    """
    # Get the number of samples in the data
    num_samples = len(data)

    # Convert hundred to a float
    hundred = 100.0

    # Get the first price in the data
    first_price = data[0]

    # Preallocate the norm_data array with uninitialized values
    norm_data = np.empty(num_samples)

    # Generate an array of x values from 1 to num_samples
    x_values = np.arange(num_samples)

    # Compute the normalized values for the data
    norm_data = np.multiply(
        np.true_divide(np.subtract(data, first_price), first_price), hundred
    )

    # Vertically stack the x_values and ones arrays
    transpose_array = np.vstack([x_values, np.ones(num_samples)])

    # Compute the least squares fit line for the data
    gradient, intercept = np.linalg.lstsq(
        transpose_array.T, data, rcond=None
    )[0]

    # Return the gradient and intercept of the least squares fit line as a tuple
    return (gradient, intercept)


def get_trades_data(mcal, rest_api, symbol, ndays, ns, verbose):
    """
    This function retrieves trade data for a given symbol using a provided mcal calendar and rest_api,
    within a specified number of days, ndays, and returns a dataframe with datetime, price, and size
    information. Optionally, it can also return the last trade price and total number of shares traded.
    It can also output summary statistics and the last few trades if verbose is set to 1.
    """
    nyse = mcal.get_calendar("NYSE")

    # Get current date and time in New York and calculate start date based on ndays
    end_dt = pd.Timestamp.now(tz="America/New_York")
    start_dt = end_dt - pd.Timedelta("%4d days" % ndays)
    _from = start_dt.strftime("%Y-%m-%d")
    to_end = end_dt.strftime("%Y-%m-%d")

    # Get the last open date within the specified time range
    nyse_schedule = (
        nyse.schedule(start_date=_from, end_date=to_end)
        .tail(1)
        .index.to_list()
    )
    last_open_date = str(nyse_schedule[0]).split(" ")[0]

    # Retrieve trade data from rest_api
    df = rest_api.get_trades(symbol, _from, last_open_date).df[
        ["price", "size"]
    ]
    total_shares_traded = df["size"].sum()

    # If ns is specified, return the last ns trades where the price changes,
    # otherwise return all trades where the price changes
    if ns > 0:
        df = df.loc[df["price"].shift() != df["price"]].tail(ns)
    else:
        df = df.loc[df["price"].shift() != df["price"]]

    # Reset index and convert timestamp to datetime in US/Central timezone
    df.reset_index(inplace=True)
    df["datetime"] = (
        pd.to_datetime(df["timestamp"])
        .dt.tz_convert("US/Central")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    # Reorder columns and set datetime as index
    df = df[["datetime", "price", "size"]]
    last_trade = df["price"].iloc[-1]
    df.set_index("datetime", inplace=True)

    # If verbose is set to 1, output summary statistics and last few trades
    if verbose == 1:
        print(
            "Minimum trade  = ", df["price"].min(), df[["price"]].idxmin()[0]
        )
        print(
            "Maximum trade  = ", df["price"].max(), df[["price"]].idxmax()[0]
        )
        print(
            "Maximum volume = ",
            df["size"].max(),
            df[["size"]].idxmax()[0],
            df["price"].iloc[df["size"].argmax()],
        )
        print("Total shares traded = ", total_shares_traded)
        print(" ")
        print(df.tail(5))

    return df, total_shares_traded, last_trade


def run(args):

    ns = int(args.ns)
    symbol = str(args.symbol.upper())
    peak_factor = float(args.factor)
    verbose = int(args.verbose)
    plot = int(args.plot)
    output = int(args.output)
    window = int(args.window)
    ndays = int(args.ndays)

    df, total_shares_traded, last_trade = get_trades_data(
        mcal, rest_api, symbol, ndays, ns, verbose
    )

    # Get the "price" column from the dataframe and convert it to a numpy array
    data = df["price"].values
    data_array = np.asarray(data, dtype=np.float32)

    # Get the number of samples in the data array
    num_samps = len(data_array)

    # Set the filter window size
    if window == 0:
        filter_window = num_samps // 20
    else:
        filter_window = window

    # Ensure that the filter window size is odd
    if (filter_window % 2) == 0:
        filter_window += 1

    # Print the filter window size if verbose mode is enabled
    if verbose == 1:
        print("Filter window = ", filter_window)

    # Apply the tolgi_filtering function to the data array
    data_filt, intercept, gradient = tolgi_filtering(
        data_array, num_samps, filter_window
    )

    # Calculate daily returns for the data array
    daily_returns = np.diff(data_array) / data_array[1:] * 100

    # Append a value of 0 to the end of the daily returns array
    daily_returns = np.append(daily_returns, 0)

    # Roll the daily returns array by 1 element
    daily_returns = np.roll(daily_returns, 1)

    # Calculate the mean and standard deviation of the data array and daily returns array
    mean = np.mean(data_array)
    std = np.std(daily_returns)

    # Calculate the delta factor
    delta_factor = std * mean * peak_factor

    # Find peaks in the filtered data using the peakdetect function
    _max, _min = peakdetect(
        data_filt, delta=delta_factor, lookahead=look_ahead
    )

    # Get the x and y values of the detected minimum and maximum points
    xmin = [p[0] for p in _min]
    ymin = [p[1] for p in _min]
    xmax = [p[0] for p in _max]
    ymax = [p[1] for p in _max]

    # Get the number of detected minimum and maximum points
    nmin = len(ymin)
    nmax = len(ymax)

    # Initialize variables for the difference, price, and action
    diff = num_samps
    price = 0.0
    action = "none"

    # If there are detected minimum and maximum points and the number of samples is greater than 0,
    # determine whether to buy or sell based on the most recent minimum or maximum point
    if (nmin > 0) & (num_samps > 0) & (nmax > 0):
        diff_min = num_samps - xmin[nmin - 1]
        diff_max = num_samps - xmax[nmax - 1]

        if diff_min < diff_max:
            action = "Buy"
            diff = diff_min
            price = ymin[nmin - 1]
        else:
            action = "Sell"
            diff = diff_max
            price = ymax[nmax - 1]

    x_values = np.linspace(1, num_samps, num_samps)

    gradient, intercept = compute_avo_attributes(data_array)

    if verbose == 1:
        print(" ")
        print(symbol, data_array[-1], gradient * 100.0, action, price, diff)

    if plot == 0:
        sys.exit()

    def thousand_comma(num):
        return "{:,d}".format(int(num))

    # Create a figure with two subplots, with the first subplot having a height
    # that is 3 times the height of the second subplot
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Plot the price data on the first subplot
    df["price"].plot(ax=ax0, alpha=0.6, linewidth=0.60)
    ax0.plot(data_filt, "yellow", linewidth=7, alpha=0.6)
    ax0.plot(gradient * x_values + intercept, "green", linewidth=8, alpha=0.6)
    ax0.plot(
        gradient * x_values + intercept,
        "lightgreen",
        linewidth=3,
        alpha=1,
        linestyle="--",
        label="Trend Prices",
    )
    ax0.plot(
        data_filt,
        "black",
        linewidth=2,
        alpha=0.9,
        linestyle="--",
        label="Filtered Prices",
    )
    ax0.scatter(
        xmin,
        df["price"].iloc[xmin],
        c="g",
        s=400,
        alpha=1,
        edgecolor="white",
        label="Buy",
    )
    ax0.scatter(
        xmax,
        df["price"].iloc[xmax],
        c="r",
        s=400,
        alpha=1,
        edgecolor="white",
        label="Sell",
    )
    ax0.grid(color="white", linestyle="--", linewidth=0.40)
    ax0.set_title(
        "Trades for: {} filter_window = {} total shares traded = {} last trade = {:.3f}".format(
            symbol,
            filter_window,
            thousand_comma(total_shares_traded),
            last_trade,
        )
    )

    # Plot the size data on the second subplot
    df["size"].plot(ax=ax1)
    ax1.grid(color="white", linestyle="--", linewidth=0.40)
    ax1.set_title("trade size for: {} (number of shares) CST".format(symbol))

    # Adjust the spacing between the subplots
    plt.tight_layout()

    # Display the figure
    plt.show()

    if output == 1:
        tfile = open("full_trades.txt", "w")
        tfile.write(df.to_csv(sep="|"))
        tfile.close()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--symbol",
        type=str,
        default="TQQQ",
        help="Stock symbol, (default=TQQQ)",
    )
    PARSER.add_argument(
        "--factor",
        type=float,
        default=0.025,
        help="Number of samples to retrieve historical trades (default=0.05)",
    )
    PARSER.add_argument(
        "--window",
        type=int,
        default=0,
        help="Number of samples in Hanning filter window (default=0, off)",
    )
    PARSER.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (default is 1=on, 0=off)",
    )
    PARSER.add_argument(
        "--plot",
        type=int,
        default=1,
        help="Plot trades (default is 1=on, 0=off)",
    )
    PARSER.add_argument(
        "--output",
        type=int,
        default=0,
        help="Write to full_trades.txt file (default is 0=off, 1=on)",
    )
    PARSER.add_argument(
        "--ndays",
        type=int,
        default=0,
        help="Number of days to retrieve historical trades (default=0, today)",
    )
    PARSER.add_argument(
        "--ns",
        type=int,
        default=0,
        help="Number of trades to process (default=0)",
    )
    ARGUMENTS = PARSER.parse_args()
    run(ARGUMENTS)
