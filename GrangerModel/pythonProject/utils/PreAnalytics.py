# checking if the sector is in up - trend or down - trend daily
# premise : if the value hasn't changed it counts as up
import matplotlib.pyplot as plt
import pandas as pd


def trend(sector):
    trend = []
    for i in range(len(sector) - 1):
        if sector[i + 1] >= sector[i]:
            trend.append(1)
        else:
            trend.append(-1)
    return trend

def weekly_table_avr(df):
    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Set the 'Date' column as the index of the DataFrame
    df.set_index('Date', inplace=True)

    # Resample the DataFrame to weekly frequency, taking the mean
    weekly_avg_oil = df['Oil'].resample('W').mean()
    weekly_avg_consumer = df['Consumer'].resample('W').mean()

    return weekly_avg_oil, weekly_avg_consumer


def graph_trend(trend_oil, trend_consumer):
    x = list(range(1, len(trend_consumer) + 1))

    fig, ax = plt.subplots(figsize=(30, 5))  # set the size of the plot
    ax.plot(x, trend_oil, 'o', markersize=5, linewidth=2, alpha=0.5, label='trend_oil')
    ax.plot(x, trend_consumer, 'o', markersize=5, linewidth=2, alpha=0.5, label='trend_consumer')
    ax.set_xlabel('time')
    ax.set_ylabel('trend')
    ax.set_title('Trend Comparison')
    ax.legend()  # add a legend to the plot
    plt.show()


def imidiate_effect(arr1, arr2):
    count = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            count += 1
    return count


def shift_1_effect(arr1, arr2):
    count = 0
    for i in range(len(arr1) - 1):
        if arr1[i + 1] == arr2[i]:
            count += 1
    return count

def shift_2_effect(arr1, arr2):
    count = 0
    for i in range(len(arr1) - 2):
        if arr1[i + 2] == arr2[i]:
            count += 1
    return count


if __name__ == '__main__':
    # todo you can also automatize this
    df_oil_consumer_1 = pd.read_csv(
        '/Users/alina.krigel/PycharmProjects/pythonProject/oil_crude_and_consumer_sector_prices 2021-01-01.csv')
    df_oil_consumer_2 = pd.read_csv(
        '/Users/alina.krigel/PycharmProjects/pythonProject/oil_crude_and_consumer_sector_prices 2021-07-01.csv')
    df_oil_consumer_3 = pd.read_csv(
        '/Users/alina.krigel/PycharmProjects/pythonProject/oil_crude_and_consumer_sector_prices 2022-01-01.csv')
    df_oil_consumer_year = pd.read_csv(
        '/Users/alina.krigel/PycharmProjects/pythonProject/oil_crude_and_consumer_sector_prices 2020-01-01.csv')

    avr_oil, avr_con = weekly_table_avr(df_oil_consumer_year)
    avr_oil_trend = trend(avr_oil)
    avr_con_trend = trend(avr_con)
    # print(len(avr_oil_trend))
    # print(len(avr_con_trend))
    print("oil")
    print(imidiate_effect(avr_oil_trend,avr_con_trend))
    print(shift_1_effect(avr_oil_trend,avr_con_trend))
    print(shift_2_effect(avr_oil_trend, avr_con_trend))
    print("consumer")
    print(imidiate_effect(avr_con_trend,avr_oil_trend))
    print(shift_1_effect(avr_con_trend,avr_oil_trend))
    print(shift_2_effect(avr_con_trend,avr_oil_trend))

    graph_trend(avr_oil_trend, avr_con_trend)

    # for i in range(3):
    #     i = i + 1
    #     if i == 1:
    #         oil = df_oil_consumer_1['Oil']
    #         consumer = df_oil_consumer_1['Consumer']
    #         trend_oil = trend(oil)
    #         trend_consumer = trend(consumer)
    #         graph_trend(trend_oil, trend_consumer)
    #     elif i == 2:
    #         oil = df_oil_consumer_2['Oil']
    #         consumer = df_oil_consumer_2['Consumer']
    #         trend_oil = trend(oil)
    #         trend_consumer = trend(consumer)
    #         graph_trend(trend_oil, trend_consumer)
    #     elif i == 3:
    #         oil = df_oil_consumer_3['Oil']
    #         consumer = df_oil_consumer_3['Consumer']
    #         trend_oil = trend(oil)
    #         trend_consumer = trend(consumer)
    #         graph_trend(trend_oil, trend_consumer)