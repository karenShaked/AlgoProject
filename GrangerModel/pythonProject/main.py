import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# TICKERS
xle_symbol = "XLE"
oil_symbol = "CL=F"

# DATES
start_date = '2010-01-01'
end_date = '2022-12-31'


def get_data_api():
    # Retrieve the stock price data using the Yahoo Finance API
    xle_df = yf.download(xle_symbol, start=start_date, end=end_date)
    oil_df = yf.download(oil_symbol, start=start_date, end=end_date)
    xle_df.to_csv('xle_prices.csv')
    oil_df.to_csv('oil_prices.csv')


def get_for_stat_data():
    xle_df = yf.download(xle_symbol, start='2023-01-01', end='2023-01-07')
    oil_df = yf.download(oil_symbol, start='2023-01-01', end='2023-01-07')
    xle_df.to_csv('xle_prices_stat.csv')
    oil_df.to_csv('oil_prices_stat.csv')


if __name__ == '__main__':
    # Plot the stock prices on the same graph
    get_data_api()
    get_for_stat_data()
    xle_df = pd.read_csv('xle_prices.csv', index_col='Date', parse_dates=True)
    oil_df = pd.read_csv('oil_prices.csv', index_col='Date', parse_dates=True)
    plt.figure(figsize=(10, 6))
    plt.title('Comparison of Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Close Price')

    # Plotting Close prices from df1
    plt.plot(xle_df.index, xle_df['Close'], label='xle')

    # Plotting Close prices from df2
    plt.plot(oil_df.index, oil_df['Close'], label='oil')

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()



