from flask import Flask, render_template, request
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)

api_key = '16XIFZ7UGMHG6GOQ'
ts = TimeSeries(key=api_key, output_format='pandas')

model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    data = data[::-1]  # Reverse the data to get it in chronological order

    # Extract the 'Open' and 'Close' prices
    dates = data.index
    open_prices = data['1. open']
    close_prices = data['4. close']

    # Visualize the data
    plt.figure(figsize=(10, 6))
    plt.plot(dates, open_prices, label='Open Price', color='blue')
    plt.plot(dates, close_prices, label='Close Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Daily Stock Prices')
    plt.legend()
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Convert the plot to base64 encoding
    plot_data = base64.b64encode(buf.getvalue()).decode()

    # Predict whether stock price will go higher or lower
    seq_length = 10
    data = data['4. close'].values
    data = data[::-1]
    X = []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
    X = np.array(X)
    predictions = model.predict(X)
    binary_predictions = (np.where(predictions > data[seq_length:], 1, 0))

    # Simulated dividends (replace with actual dividend predictions)
    dividends = [0.5, 0.6, 0.7, 0.8, 0.9]  # Example: Simulated dividends

    return render_template('prediction.html', ticker=ticker, plot_data=plot_data, binary_predictions=binary_predictions, dividends=dividends)

if __name__ == '__main__':
    app.run(debug=True)
