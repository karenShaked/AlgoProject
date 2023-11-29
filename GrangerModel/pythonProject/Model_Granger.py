import seaborn as sns
import tkinter as tk
from tkinter import ttk, scrolledtext
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
import statistics

# TICKERS
xle_symbol = "XLE"
oil_symbol = "CL=F"
returns = []

# DATES
start_date = '2012-01-01'
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


def fit_arima_and_append(df, order=(1, 1, 1)):
    """
    Fit an ARIMA model to the 'Close' column in the DataFrame and append the results as a new column named 'ARIMA'.
"""
    new_df = df.copy()
    new_df.index = pd.date_range(start=new_df.index[0], periods=len(new_df), freq=None)

    # Fit ARIMA model
    model = ARIMA(new_df['Close'], order=order)
    model_fit = model.fit()

    # Get predictions
    predictions = model_fit.predict(start=new_df['Close'].index[0], end=new_df['Close'].index[-1])

    # remove additional noise
    predictions = predictions.rolling(window=30).mean()

    # Append predictions as 'ARIMA' column
    df['ARIMA'] = predictions

    return df


def granger_test(xle, oil, _start):
    period_ = 100
    max_lag_ = 21
    combined_data = combine_data_between_dates(_start, period_, xle, oil)
    granger_df = granger_causality_test(combined_data.copy(), max_lag_)
    granger_df = combine_dates_reverse(combined_data, _start, granger_df)
    granger_df = set_lags(granger_df, _start)
    return granger_df


def combine_data_between_dates(_end_date_, _period, energy, oil):
    """
    Combine two DataFrames or Series within a specified date range.
    Returns
    -------
    pandas DataFrame
        Combined DataFrame of the two input data within the specified date range.
    """
    _end_date = pd.to_datetime(_end_date_)
    _start_date = _end_date - pd.DateOffset(days=_period-1)
    combined_df = pd.concat([energy.loc[_start_date:_end_date], oil.loc[_start_date:_end_date]], axis=1)
    combined_df.dropna(inplace=True)
    columns_to_keep = ['ARIMA']
    combined_df = combined_df.loc[:, columns_to_keep]
    return combined_df


def granger_causality_test(data, max_lag):
    """
    Apply the Granger causality test to detect whether oil price is affecting energy prices.

    (a low p-value indicates that there is a statistically significant relationship
     between the variables being tested, supporting the presence of Granger causality.)
    Returns
        Results of the Granger causality test.
    """
    data = data.set_index(pd.Index(range(len(data))))
    results = grangercausalitytests(data, max_lag, verbose=False)
    results_df = print_granger_results(results)
    print_top_lags(results)
    return results_df


def print_granger_results(granger_results):
    lag_orders = sorted(granger_results.keys())
    columns = [
        'Lag Order',
        'F-Statistic',
        'F-P-Value',
        'Chi-Squared Statistic',
        'Chi-Squared P-Value',
        'Likelihood Ratio Statistic',
        'Likelihood Ratio P-Value'
    ]

    rows = []
    for lag_order in lag_orders:
        result = granger_results[lag_order]
        f_statistic = result[0]['ssr_ftest']
        chi_squared = result[0]['ssr_chi2test']
        lr_statistic = result[0]['lrtest']

        row = [
            lag_order,
            f_statistic[0],
            f_statistic[1],
            chi_squared[0],
            chi_squared[1],
            lr_statistic[0],
            lr_statistic[1]
        ]
        rows.append(row)

    _df = pd.DataFrame(rows, columns=columns)
    print(_df.to_string(index=False))
    return _df


def get_top_lags(granger_results):
    lag_orders = sorted(granger_results.keys())

    # Sort lag orders based on the p-values in ascending order
    sorted_lags = sorted(lag_orders, key=lambda lag: granger_results[lag][0]['params_ftest'][1])

    # Get the top 3 lag orders with the lowest p-values
    top_lags = sorted_lags[:3]

    return top_lags


def print_top_lags(granger_results):
    top_lags = get_top_lags(granger_results)

    print("Top 3 Lag Orders:")
    print("-----------------")
    for lag in top_lags:
        p_value = granger_results[lag][0]['params_ftest'][1]
        print(f"Lag Order: {lag}, p-value: {p_value}")


def combine_dates_reverse(original_df, curr_date, results_df):
    """
    Combine the dates columns of the original DataFrame to the results DataFrame
    in reverse order based on the current date.
        The results DataFrame with the dates columns combined in reverse order.
    """
    # Convert the curr_date to pandas datetime if it's not already
    if not isinstance(curr_date, pd.Timestamp):
        curr_date = pd.to_datetime(curr_date)

    # Get the dates from the original_df up to the curr_date in reverse order
    dates = original_df.index[original_df.index < curr_date][::-1][:len(results_df)]

    # Combine the dates columns in the results_df in reverse order
    results_df['Date'] = dates

    return results_df


def set_lags(granger_df, start_date_):
    start_date_ = pd.to_datetime(start_date_)  # Specify your desired start date
    granger_df['Lag Order'] = (start_date_ - granger_df['Date']).dt.days
    return granger_df


def assign_p_values_scores(df_granger):
    scores = []

    for p_value in df_granger['F-P-Value']:
        if p_value < 0.01:
            scores.append(100)
        elif 0.01 <= p_value < 0.05:
            scores.append(np.interp(p_value, [0.01, 0.05], [90, 100]))
        elif 0.05 <= p_value < 0.1:
            scores.append(np.interp(p_value, [0.05, 0.1], [80, 90]))
        elif 0.1 <= p_value < 0.2:
            scores.append(np.interp(p_value, [0.1, 0.2], [70, 80]))
        elif 0.2 <= p_value < 0.3:
            scores.append(np.interp(p_value, [0.2, 0.3], [60, 70]))
        elif 0.3 <= p_value < 0.4:
            scores.append(np.interp(p_value, [0.3, 0.4], [50, 60]))
        elif 0.4 <= p_value <= 0.5:
            scores.append(np.interp(p_value, [0.4, 0.5], [40, 50]))
        else:
            scores.append(np.interp(p_value, [0.5, 1], [0, 40]))

    df_granger['p-value_scores'] = scores

    return df_granger


def assign_change_oil(df_oil, curr_date, df_granger):
    change_column = []
    curr_date = pd.to_datetime(curr_date)
    for lag in df_granger['Lag Order']:
        pct_df = df_oil['Close'].pct_change(periods=lag)
        change = pct_df[pct_df.index >= curr_date]
        change = change[0]
        change_column.append(change.item()*100)
    df_granger['change_oil_score'] = change_column
    return df_granger


def assign_change_xle(df_xle, df_granger, _date):
    xle_change = df_xle.loc[_date, 'Close']
    df_granger['change_XLE_score'] = xle_change
    return df_granger


def p_value_change_pct_graph_print(df_granger):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the p-value scores
    ax1.plot(df_granger['Lag Order'], df_granger['p-value_scores'], marker='o')
    ax1.set_xlabel('Lag Order')
    ax1.set_ylabel('p-value')
    ax1.set_title('Statistical Analysis: p-value Scores')

    # Plot the change in oil prices
    ax2.plot(df_granger['Lag Order'], df_granger['change_oil_score'], marker='o')
    ax2.set_xlabel('Lag Order')
    ax2.set_ylabel('Change in Oil (%)')
    ax2.set_title('Change in Oil Prices')

    # Adjust the layout
    fig.tight_layout()

    # Display the graphs
    plt.show()


def assign_final_score(df_granger):
    """integrate between the p-value which indicates the statistical significance
    of the Granger causality test. and  the behavior of the oil prices
     """
    short_or_long = []
    final_score = []
    for index, row in df_granger.iterrows():
        oil_change = row['change_oil_score']
        _xle_change = row['change_XLE_score']
        _p_val_score = row['p-value_scores']
        _lag = row['Lag Order']
        if oil_change < 0:
            short_or_long.append("short")
            oil_change = oil_change*-1
            _xle_change = _xle_change*-1
        else:
            if oil_change > 0:
                short_or_long.append("long")
            else:
                short_or_long.append("none")
        final_score.append(oil_change*0.25 + _xle_change*0.25 + _p_val_score*0.5 + _lag*0.2)
    df_granger["short_or_long"] = short_or_long
    df_granger["final score"] = final_score
    return df_granger


def final_score_graph_print(df_granger):
    print_graph(df_granger, 'layout of the final scores', 'lags', 'final score',
                'Lag Order', 'final score', 'final score')


def buy_based_on_final_score(df_granger):
    bar = 50
    max_score = df_granger["final score"].max()
    max_score_row = df_granger.loc[df_granger["final score"] == max_score, :]
    if max_score > bar and max_score_row["short_or_long"].values[0] != "none":
        return_val = [max_score_row, "BUY"]
        return return_val
    return_val = [max_score_row, "DONT BUY"]
    return return_val


def print_two_graphs(df1, df2, title, x_label, y_label, y_column_name_df1, y_column_name_df2,
                     x_column_name_df1, x_column_name_df2, df1_values_name, df2_values_name):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_column_name_df1 == "":
        x_column_name_df1 = df1.index
    else:
        x_column_name_df1 = df1[x_column_name_df1]

    if x_column_name_df2 == "":
        x_column_name_df2 = df2.index
    else:
        x_column_name_df2 = df2[x_column_name_df2]

    # Plotting values from df1
    plt.plot(x_column_name_df1, df1[y_column_name_df1], label=df1_values_name)

    # Plotting values from df2
    plt.plot(x_column_name_df2, df2[y_column_name_df2], label=df2_values_name)

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


def print_graph(df1, title, x_label, y_label, x_column_name, y_column_name, df1_name):

    if x_column_name == "":
        x_column_name = df1.index
    else:
        x_column_name = df1[x_column_name]

    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Plotting values from df1
    plt.plot(x_column_name, df1[y_column_name], label=df1_name, color='purple', linewidth=2)

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    # Customize tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add a legend with a border and a shadow
    legend = plt.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='black')
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.8)

    # Show the plot
    plt.tight_layout()
    plt.show()


def popping_window_describe(df):
    # Assuming your DataFrame is named 'df'
    df = pd.DataFrame(df.describe())
    df = df.reset_index()

    # Create a new Tkinter window
    window = tk.Tk()

    # Configure custom styles for the Treeview widget
    style = ttk.Style()
    style.theme_use("default")
    style.configure("Treeview",
                    background="white",
                    foreground="black",
                    fieldbackground="white")
    style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))

    # Create a Treeview widget to display the DataFrame as a table
    tree = ttk.Treeview(window, style="Treeview")
    tree.grid(row=0, column=0, sticky="nsew")

    # Configure the Treeview columns based on the DataFrame columns
    tree["columns"] = list(df.columns)

    # Configure column headings
    for column in df.columns:
        tree.heading(column, text=column)

    # Insert DataFrame rows into the Treeview widget
    for row in df.itertuples(index=False):
        tree.insert("", tk.END, values=row)

    # Configure resizing behavior of columns and rows
    for column in df.columns:
        tree.column(column, anchor="center", width=100)
        tree.heading(column, anchor="center")

    window.mainloop()


def print_to_buy(to_buy):
    # display row
    print("max score row:")
    print(to_buy[0])

    # Display the label
    print("\nLabel:", to_buy[1])


def run_granger(_clean_xle, _clean_oil, _date, _oil_df, _xle_change_30):
    result_df = granger_test(_clean_xle, _clean_oil, _date)
    result_df = assign_p_values_scores(result_df)
    result_df = assign_change_oil(_oil_df, _date, result_df)
    result_df = assign_change_xle(_xle_change_30, result_df, _date)
    p_value_change_pct_graph_print(result_df)
    assign_final_score(result_df)
    popping_window_describe(result_df)
    final_score_graph_print(result_df)
    to_buy_ = buy_based_on_final_score(result_df)
    print_to_buy(to_buy_)
    return to_buy_


def buy_long(buy_row, _oil_df, final_date):

    # get lag and initial date
    lag = int(buy_row['Lag Order'][0])
    initial_oil_date = buy_row['Date'][0]

    # Last date
    last_date = initial_oil_date + pd.DateOffset(days=lag)

    # Filter the DataFrame based on the date range
    filtered_df = _oil_df.loc[(_oil_df.index >= initial_oil_date) &
                              (_oil_df.index <= last_date)]

    # Find the row with the maximum 'Close' value
    max_row = filtered_df.loc[filtered_df['Close'].idxmax()]

    # Get the index (date) of the row with the maximum 'Close' value
    max_date = max_row.name

    # Find the last row
    last_row = filtered_df.tail(1)

    while (max_row['Close'] == last_row['Close'][0]) and (last_date < final_date):
        last_date = _oil_df.index[_oil_df.index > last_date].min()

        # Filter the DataFrame based on the date range
        filtered_df = _oil_df.loc[(_oil_df.index >= initial_oil_date) &
                                  (_oil_df.index <= last_date)]

        # Find the row with the maximum 'Close' value
        max_close_row = filtered_df.loc[filtered_df['Close'].idxmax()]

        # Get the index (date) of the row with the maximum 'Close' value
        max_date = max_close_row.name

        last_row = filtered_df.tail(1)
    days_till_sell = max_date - initial_oil_date
    return days_till_sell.days


def buy_short(buy_row, _oil_df, final_date):

    # get lag and initial date
    lag = int(buy_row['Lag Order'][0])
    initial_oil_date = buy_row['Date'][0]

    # Last date
    last_date = initial_oil_date + pd.DateOffset(days=lag)

    # Filter the DataFrame based on the date range
    filtered_df = _oil_df.loc[(_oil_df.index >= initial_oil_date) &
                              (_oil_df.index <= last_date)]

    # Find the row with the min 'Close' value
    min_row = filtered_df.loc[filtered_df['Close'].idxmin()]

    # Get the index (date) of the row with the min 'Close' value
    min_date = min_row.name

    # Find the last row
    last_row = filtered_df.tail(1)

    while (min_row['Close'] == last_row['Close'][0]) and (last_date < final_date):
        last_date = _oil_df.index[_oil_df.index > last_date].min()

        # Filter the DataFrame based on the date range
        filtered_df = _oil_df.loc[(_oil_df.index >= initial_oil_date) &
                                  (_oil_df.index <= last_date)]

        # Find the row with the min 'Close' value
        min_close_row = filtered_df.loc[filtered_df['Close'].idxmin()]

        # Get the index (date) of the row with the min 'Close' value
        min_date = min_close_row.name

        last_row = filtered_df.tail(1)
    days_till_sell = min_date - initial_oil_date
    return days_till_sell.days


def demo_portfolio(_clean_xle, _clean_oil, curr_date, _oil_df, _xle_change_30, _xle_df,
                   period=45, _cash=1000):
    """Uses the algorithm to determine in a period of half a year
    whether to buy the energy sector stock (for short/long) or not (based on final scores).
     if the algorithm decides to buy it - the algorithm calculates best time to sell based
     on the good correlation found .

     Returns
     _______
     the cash we have after half a year
     """
    # reassure its date instance
    if not isinstance(curr_date, pd.Timestamp):
        curr_date = pd.to_datetime(curr_date)

    # calculate end date
    end_date_ = curr_date + pd.DateOffset(days=period-1)

    while curr_date < end_date_:
        to_buy_row = run_granger(_clean_xle, _clean_oil, curr_date, _oil_df, _xle_change_30)
        if to_buy_row[1] == "BUY":
            # days_till_sell = 0
            buy_row = to_buy_row[0]
            buy_row = buy_row.reset_index()
            if buy_row['short_or_long'][0] == "long":
                days_till_sell = buy_long(buy_row, _oil_df, end_date_)
                sell_date = curr_date + pd.DateOffset(days=days_till_sell)
            else:
                days_till_sell = buy_short(buy_row, _oil_df, end_date_)
                sell_date = curr_date + pd.DateOffset(days=days_till_sell)
                sell_date = _xle_df.index[_xle_df.index >= sell_date].min()
            _cash = sell_stock(_xle_df, _cash, curr_date, sell_date, buy_row['short_or_long'][0])
            curr_date = sell_date

        else:
            curr_date = _xle_df.index[_xle_df.index > curr_date].min()
        if not isinstance(curr_date, pd.Timestamp):
            curr_date = pd.to_datetime(curr_date)
    return _cash


def sell_stock(_xle_df, cash, curr_date, sell_date, short_or_long, risk_free_rate=0.05/365):

    # Calculates the buy and sell prices
    sell_date = _xle_df.index[_xle_df.index > sell_date].min()
    close_price_buy = _xle_df.loc[curr_date, 'Close']
    close_price_sell = _xle_df.loc[sell_date, 'Close']

    # Calculate the percentage change
    percentage_change = (close_price_sell - close_price_buy) / close_price_buy * 100
    if short_or_long == "short":
        percentage_change = percentage_change * -1

    """daily_returns = _xle_df.loc[(_xle_df.index >= curr_date) & (_xle_df.index <= sell_date)].pct_change()
    daily_returns = daily_returns.dropna()
    avg_daily_return = daily_returns['Close'].mean()
    std_daily_return = daily_returns['Close'].std()
    sharpe_ratio = (avg_daily_return - risk_free_rate) / std_daily_return"""
    returns.append(percentage_change)
    final_value = cash*(1+(percentage_change/100))
    profit = final_value-cash

    summary = f"Buy Date: {curr_date}\n"
    summary += f"Sell Date: {sell_date}\n"
    summary += f"Initial Value: {cash}\n"
    summary += f"Final Value: {final_value}\n"
    summary += f"Profit: {profit}\n"
    summary += f"type: {short_or_long}\n"
    summary += f"change: {percentage_change}\n"
    # summary += f"Sharpe Ratio: {sharpe_ratio}"
    print(summary)
    return final_value


if __name__ == '__main__':
    # Plot the stock prices on the same graph
    get_data_api()
    get_for_stat_data()
    xle_df = pd.read_csv('xle_prices.csv', index_col='Date', parse_dates=True)
    oil_df = pd.read_csv('oil_prices.csv', index_col='Date', parse_dates=True)

    # Apply ARIMA to the stock price data
    clean_xle = fit_arima_and_append(xle_df)
    clean_oil = fit_arima_and_append(oil_df)

    # drop null
    clean_xle = clean_xle.dropna()
    clean_oil = clean_oil.dropna()

    xle_change_30 = xle_df.pct_change(periods=30)

    print_two_graphs(clean_xle, clean_oil, "arima oil & xle", 'Date', 'Price', 'ARIMA', 'ARIMA',
                     "", "", 'xle', 'oil')

    # Choose date
    _date = '2013-11-11'

    # Demo portfolio
    final_value = demo_portfolio(clean_xle, clean_oil, _date, oil_df, xle_change_30, xle_df)
    _date = pd.Timestamp('2013-11-11')
    sell_date = _date + pd.DateOffset(days=45)
    avg_daily_return = statistics.mean(returns)
    std_daily_return = statistics.stdev(returns)
    risk_free_rate = 0.55
    sharpe_ratio = (avg_daily_return - risk_free_rate) / std_daily_return
    profit = final_value - 1000
    summary = f"Buy Date: {_date}\n"
    summary += f"Sell Date: {sell_date}\n"
    summary += f"Initial Value: 1000\n"
    summary += f"Final Value: {final_value}\n"
    summary += f"Profit: {profit}\n"
    summary += f"Sharpe Ratio: {sharpe_ratio}"
    print(summary)




