# Stock Price Predictor
This Stock Price Predictor program uses TensorFlow, NumPy, Pandas, Matplotlib, and yfinance libraries to predict stock prices using long short-term memory (LSTM) neural networks.

It defines a class called StockPricePredictor with attributes such as the ticker symbol, prediction days, past prediction period, and future prediction period.

The program first retrieves historical stock data from Yahoo Finance API and creates the training input and output data. It then compiles the LSTM model with dropout layers and trains on the training data, using an early stopping callback to prevent overfitting to the training set.

The early stopping callback monitors the validation loss during training and stops the training process if the validation loss doesn't improve after a certain number of epochs.

Finally, the program uses the trained model to predict stock prices and plots the actual and predicted prices to visualize the results.

# Examples of the Stock Price Predictor predictions
![S P_500](https://user-images.githubusercontent.com/108039068/233427715-dbaff19f-a098-4c91-b639-dfcb4b3abbf5.png)
![Coca_Cola](https://user-images.githubusercontent.com/108039068/233427728-ba9b72ff-cb32-43a5-b8fd-3d87ad72e64b.png)
![Berkshire_Hathaway](https://user-images.githubusercontent.com/108039068/233427745-5b36145f-c330-4423-b302-b704a9f6c2f2.png)
