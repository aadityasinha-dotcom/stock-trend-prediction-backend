from django.shortcuts import render
from .serializers import StudentSerializer
from rest_framework.generics import ListAPIView
from .models import Student
import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import json
from django.http import JsonResponse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Create your views here.
# class StudentList(ListAPIView):
    # queryset = Student.objects.all()
    # serializer_class = StudentSerializer
def closing_graph(df, ticker):
    plt.plot(df['Close'])
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'{ticker} Close Price')
    plt.savefig(f'media/close/{ticker}.png')

def moving_average(df, ticker):
    ma_day = [10, 20, 50]
    columns = {}
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        columns[column_name] = df['Adj Close'].rolling(ma).mean()
    plt.plot(df['Adj Close'])
    plt.plot(columns["MA for 10 days"], 'r')
    plt.plot(columns["MA for 20 days"], 'g')
    plt.plot(columns["MA for 50 days"], 'o')
    plt.title("Moving average for 10, 20, 50 days")
    plt.savefig(f'media/ma/{ticker}.png')

def adjacent_close(df, ticker):
    daily_return = df['Adj Close'].pct_change()
    plt.plot(daily_return, linestyle='--', marker='o')
    plt.title("Daily Return")
    plt.savefig(f'media/adj_close/{ticker}.png')
    # Daily Return
    hist = daily_return.hist(bins=50)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title("Daily Return")
    plt.savefig(f'media/daily_return/{ticker}.png')

def closing_price_history(df, ticker):
    plt.figure(figsize=(16,6))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.savefig(f'media/closing_price_history/{ticker}.png')

def train_model(df, ticker):
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))
    # Scale Data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    # Create the training data set 
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002 
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Build the model

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=10)

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2))) * 10
    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Original Price', 'Val', 'Predicted Price'])
    plt.savefig(f'media/prediction/{ticker}.png')

    return rmse



def display_stock(request):
    # if request.method == 'POST':
        ticker = request.GET.get('ticker')
        end = datetime.now()
        start = datetime(end.year - 1, end.month, end.day)
        stock = yf.download(ticker, start, end)
        df = pd.DataFrame(stock)
        # Closing Graph
        closing_graph(df, ticker)
        # Moving Average
        moving_average(df, ticker)
        # Adjacent Close
        adjacent_close(df, ticker)
        # Closing Price History
        closing_price_history(df, ticker)
        # Train Data Model
        rmse = train_model(df, ticker)
        stock_json = json.loads(stock.to_json())
        stock_json['file_name'] = f"{ticker}.png"
        stock_json['rmse'] = f"{rmse}"
        return JsonResponse(stock_json, safe=False)
        # return render(request, 'stock.html', context)
    # else:
        # return render(request, 'input.html')

