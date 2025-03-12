# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the saved model and scaler
def load_models():
    with open('stock_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_Tesla.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()

@app.route('/')
def home():
    return render_template('index_mlstockprice.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        open_price = float(request.form['open'])
        high_price = float(request.form['high'])
        low_price = float(request.form['low'])
        volume = float(request.form['volume'])

        # Make prediction
        input_data = np.array([[open_price, high_price, low_price, volume]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            'success': True,
            'prediction': f'${prediction:.2f}',
            'input_data': {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'volume': volume
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
