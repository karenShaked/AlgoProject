import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load OHLCH dataframe
    oil_df = pd.read_csv('oil_prices.csv')

    # Create new column for previous day's closing price
    oil_df['PrevClose'] = oil_df['Close'].shift(1)

    # Remove first row
    oil_df = oil_df.dropna()

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(oil_df['PrevClose'], oil_df['Close'], test_size=0.2, random_state=42)

    # Fit linear regression model
    lr = LinearRegression()
    lr.fit(X_train.values.reshape(-1, 1), y_train)

    # Make predictions on testing data
    y_pred = lr.predict(oil_df['Close'].values.reshape(-1, 1))
    # Evaluate performance using mean absolute error
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # plot :
    fig, ax = plt.subplots(figsize=(12, 6))

    ax2 = ax.twinx()
    ax2.plot(oil_df.index, oil_df['Close'], color='red', label='Crude Oil Futures (CL=F)')
    ax2.set_ylabel('Price ($)')
    ax2.legend(loc='upper left')
    plt.plot(oil_df.index, y_pred, color='blue')
    fig.suptitle('Historical Prices for Energy Sector and Crude Oil Futures', fontsize=16, y=1.05)
    plt.show()


