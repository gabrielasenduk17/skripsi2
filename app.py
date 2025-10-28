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
arima_model = joblib.load("model/arima_model_predictionmodel.pkl")
lstm_model = load_model("model/hybrid_lstm_predictionmodel.h5", compile=False)

scaler = joblib.load("model/scaler_adj_close_predictionmodel.save")
scaler_residual = joblib.load("model/scaler_residual_predictionmodel.save")

LOOK_BACK = 60 # Pastikan 60 sesuai dengan yang digunakan saat training

# ===== Prediction Function (Koreksi Final) =====
def predict_stocks(df, n_forecast=60, look_back=LOOK_BACK):
    data = df["adj_close"].values

    # --- 1. ARIMA Forecast (Prediksi Tren Linear) ---
    arima_forecast = arima_model.forecast(steps=n_forecast)

    # --- 2. LSTM Forecast (Prediksi Residual/Error) ---
    last_window_data = data[-look_back:]
    
    # Skala input menggunakan **scaler_residual** (scaler yang dilatih pada residual)
    scaled_input_resid = scaler_residual.transform(last_window_data.reshape(-1, 1))
    
    # Siapkan input untuk LSTM
    cur_input_resid = scaled_input_resid.reshape(1, look_back, 1)
    forecast_resid_scaled = []

    # Prediksi berulang
    for _ in range(n_forecast):
        pred_resid_scaled = lstm_model.predict(cur_input_resid, verbose=0)[0][0]
        forecast_resid_scaled.append(pred_resid_scaled)
        cur_input_resid = np.append(cur_input_resid[:, 1:, :], [[[pred_resid_scaled]]], axis=1)

    # Inverse transform menggunakan **scaler_residual**
    forecast_resid = scaler_residual.inverse_transform(np.array(forecast_resid_scaled).reshape(-1, 1)).flatten()

    # --- 3. Hybrid Forecast (Gabungan ARIMA + Residual) ---
    # Logika Hibrida yang benar: ARIMA (Tren) + Residual LSTM (Error Non-linear)
    hybrid_forecast = arima_forecast + forecast_resid
    
    # --- 4. Prediksi LSTM Harga Langsung (Hanya untuk Plot Garis Hijau) ---
    # Dipertahankan untuk plot 'LSTM Forecast' (Garis hijau)
    scaled_data_price = scaler.transform(data.reshape(-1, 1)) # Gunakan scaler harga
    cur_input_price = scaled_data_price[-look_back:].reshape(1, look_back, 1)
    forecast_lstm_price_scaled = []
    
    for _ in range(n_forecast):
        pred_price = lstm_model.predict(cur_input_price, verbose=0)[0][0]
        forecast_lstm_price_scaled.append(pred_price)
        cur_input_price = np.append(cur_input_price[:, 1:, :], [[[pred_price]]], axis=1)

    forecast_lstm_price = scaler.inverse_transform(np.array(forecast_lstm_price_scaled).reshape(-1, 1)).flatten()

    return data, arima_forecast, forecast_lstm_price, hybrid_forecast


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