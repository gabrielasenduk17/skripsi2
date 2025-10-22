from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, joblib

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model saat server dijalankan
arima_model = joblib.load("model/arima_model.pkl")
lstm_model = load_model("model/hybrid_lstm_model.h5", compile=False)

# ===== Prediction Function =====
def predict_stocks(df, n_forecast=30):
    data = df["adj_close"].values

    # --- ARIMA Forecast ---
    arima_forecast = arima_model.forecast(steps=n_forecast)

    # --- LSTM Forecast ---
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data.reshape(-1,1))
    window = 10
    last_window = scaled_data[-window:]
    cur_input = last_window.reshape(1, window, 1)
    forecast_lstm = []

    for _ in range(n_forecast):
        pred = lstm_model.predict(cur_input, verbose=0)[0][0]
        forecast_lstm.append(pred)
        cur_input = np.append(cur_input[:,1:,:], [[[pred]]], axis=1)

    forecast_lstm = scaler.inverse_transform(np.array(forecast_lstm).reshape(-1,1)).flatten()

    # --- Hybrid Forecast ---
    hybrid_forecast = (arima_forecast + forecast_lstm) / 2.0

    return data, arima_forecast, forecast_lstm, hybrid_forecast


# ===== Plot Function =====
def plot_results(history, arima_forecast, lstm_forecast, hybrid_forecast):
    plt.figure(figsize=(10,6))
    plt.plot(history, label="History")
    plt.plot(range(len(history), len(history)+len(arima_forecast)), arima_forecast, "r--", label="ARIMA Forecast")
    plt.plot(range(len(history), len(history)+len(lstm_forecast)), lstm_forecast, "g--", label="LSTM Forecast")
    plt.plot(range(len(history), len(history)+len(hybrid_forecast)), hybrid_forecast, "b", label="Hybrid Forecast", linewidth=2)
    plt.legend()
    plt.title("Hybrid ARIMA-LSTM Stock Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()
    return img


# ===== Wrapper untuk pengecekan kolom =====
def predict_stock(df, n_forecast=60):
    # Cari kolom harga penutupan
    possible_cols = ["adj_close", "Adj Close", "close", "Close"]
    col_name = None
    for c in possible_cols:
        if c in df.columns:
            col_name = c
            break
    if col_name is None:
        raise ValueError(f"Kolom harga penutupan tidak ditemukan. Kolom yang tersedia: {list(df.columns)}")

    # Ubah nama kolom agar seragam
    df = df.rename(columns={col_name: "adj_close"})

    # Jalankan fungsi prediksi utama
    history, arima_forecast, lstm_forecast, hybrid_forecast = predict_stocks(df, n_forecast)

    # Return hasil prediksi
    return history, arima_forecast, lstm_forecast, hybrid_forecast


# ===== Routes =====
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    df = pd.read_csv(file)
    history, arima_f, lstm_f, hybrid_f = predict_stock(df)
    img = plot_results(history, arima_f, lstm_f, hybrid_f)
    return send_file(img, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
