# Importing libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yahoo_fin
import datetime
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler

# Override the pandas_datareader method to use yfinance
yahoo_fin.pdr_override()

# Define a class for stock price prediction
class StockPricePredictor:
    """
    A class for predicting the stock price of a given ticker symbol using a Long Short-Term Memory (LSTM) neural network.

    Attributes:
    - ticker_symbol (str): The ticker symbol of the stock to be predicted
    - prediction_days (int): The number of days in the past to be used as input for predicting each day's price
    - past_prediction_time_period (int): The number of days in the past to be used for training the model
    - future_prediction_time_period (int): The number of days in the future to predict prices for

    Methods:
    - predict_stock_prices(): Predict the future stock prices based on the model and input data
    """

    def __init__(self, ticker_symbol, prediction_days=365, past_prediction_time_period=3650, future_prediction_time_period=365):
        """
        Initializes the StockPricePredictor class.

        Parameters:
        - ticker_symbol (str): A string representing the ticker symbol of the stock to be predicted.
        - prediction_days (int): An integer representing the number of days to be predicted (default is 365).
        - past_prediction_time_period (int): An integer representing the number of days in the past from today to be considered for prediction (default is 3650).
        - future_prediction_time_period (int): An integer representing the number of days in the future from today to be considered for prediction (default is 365).
        """

        self.ticker_symbol = ticker_symbol
        self.prediction_days = prediction_days
        self.past_prediction_time_period = past_prediction_time_period
        self.future_prediction_time_period = future_prediction_time_period
        
        # Get the stock name from Yahoo Finance API
        self.stock_name = yahoo_fin.Ticker(ticker_symbol).info['longName']
        
        # Create an early stopping callback for model training
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
        
        # Initialize lists for training and testing data
        self.train_inputs = []
        self.train_outputs = []
        self.test_inputs = []
        
        # Get historical stock data for the given ticker symbol and time period from Yahoo Finance API
        self.train_data = pdr.get_data_yahoo(self.ticker_symbol, start='2000-01-01', end=(datetime.date.today() - datetime.timedelta(days=self.past_prediction_time_period)))

        # Scale the stock data to be between 0 and 1
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaled_data = self.scaler.fit_transform(self.train_data['Adj Close'].values.reshape(-1,1))

        # Create the training input and output data
        for day in range(self.prediction_days, len(self.scaled_data)):
            self.train_inputs.append(self.scaled_data[day-self.prediction_days:day, 0])
            self.train_outputs.append(self.scaled_data[day, 0])

        # Convert training inputs and outputs to numpy arrays and reshape training inputs
        self.train_inputs, self.train_outputs = np.array(self.train_inputs), np.array(self.train_outputs)
        self.train_inputs = np.reshape(self.train_inputs, (self.train_inputs.shape[0], self.train_inputs.shape[1], 1))

        # Create and compile the LSTM model for stock price prediction
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(self.train_inputs.shape[1], 1)))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.LSTM(units=50))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the LSTM model on the training data
        self.model.fit(self.train_inputs, self.train_outputs, epochs=300, batch_size=128, validation_split=0.2, callbacks=[self.early_stopping])

        # Initialize lists for true prices, combined data, model inputs, and predicted prices
        self.all_true_prices = []
        self.combined_data = []
        self.model_inputs = []
        self.past_predicted_prices = []

    def predict_stock_prices(self):
        """
        Predicts the stock prices for a given ticker symbol using a trained model and plots the true and predicted prices.

        Parameters:
        - self (object): Instance of the StockPricePredictor class
        """
        # Get historical stock data for the given ticker symbol and past past_prediction_time_period days from Yahoo Finance API
        test_data = pdr.get_data_yahoo(self.ticker_symbol, start=(datetime.date.today() - datetime.timedelta(days=self.past_prediction_time_period)), end=datetime.date.today())

        # Set the true_prices variable to the Adjusted Close values of the test_data DataFrame
        self.all_true_prices = test_data['Adj Close'].values

        # Concatenate the Adjusted Close values of the training and test data and store it in the combined_data variable
        self.combined_data = pd.concat((self.train_data['Adj Close'], test_data['Adj Close']), axis=0)

        # Scale the combined_data using the scaler object and store it in the model_inputs variable
        self.model_inputs = self.scaler.transform(self.combined_data[len(self.combined_data) - len(test_data) - self.prediction_days:].values.reshape(-1, 1))

        # Create a 3D numpy array of shape (n_samples, n_timesteps, n_features) for the test_inputs
        for day in range(self.prediction_days, len(self.model_inputs)):
            self.test_inputs.append(self.model_inputs[day-self.prediction_days:day,0])
        self.test_inputs = np.array(self.test_inputs)
        self.test_inputs = np.reshape(self.test_inputs, (self.test_inputs.shape[0], self.test_inputs.shape[1], 1))
        
        # Use the trained model to predict the stock prices for the test_inputs and unscale the predicted prices using the scaler object
        self.past_predicted_prices = self.scaler.inverse_transform(self.model.predict(self.test_inputs))

        # Generate the inputs for the future predictions
        future_inputs = []
        for day in range(len(self.model_inputs) - self.prediction_days, len(self.model_inputs)):
            future_inputs.append(self.model_inputs[day-self.prediction_days:day,0])
        future_inputs = np.array(future_inputs)
        future_inputs = np.reshape(future_inputs, (future_inputs.shape[0], future_inputs.shape[1], 1))

        # Use the trained model to predict the future stock prices, unscale the predicted prices using the scaler object and combine the predicted prices for the past, and predicted prices for the future
        all_predicted_prices = np.concatenate([self.past_predicted_prices, self.scaler.inverse_transform(self.model.predict(future_inputs))])

        # Plot the true and predicted prices on a graph using matplotlib
        plt.plot(self.all_true_prices, color='red', label='True Price')
        plt.plot(all_predicted_prices, color='blue', label='Predicted Price')
        plt.title(f'{self.stock_name} True vs. Predicted Share Price over last {self.past_prediction_time_period} days and {self.future_prediction_time_period} days in the future ({datetime.datetime.today().strftime("%B %d, %Y")})', fontsize=10)
        plt.xlabel('Days')
        plt.ylabel('Share Price')
        plt.legend()
        plt.show()

# Checks if the script is being run directly or imported as a module
if __name__ == '__main__':
    # Initialize ticker symbol of desired stock
    ticker_symbol = '^GSPC'

    # Create an instance of the StockPricePredictor class and call the predict_stock_prices method to predict stock prices
    predicted_prices = StockPricePredictor(ticker_symbol).predict_stock_prices()