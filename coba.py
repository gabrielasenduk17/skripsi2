from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, joblib

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# === Load Model Hybrid ===
arima_model = joblib.load("model/arima_model.pkl")
lstm_model = load_model("model/hybrid_lstm_model.h5", compile=False)


# === Fungsi Prediksi ===
def predict_stocks(df, n_forecast=30):
    data = df["adj_close"].values

    # --- ARIMA Forecast ---
    arima_forecast = arima_model.forecast(steps=n_forecast)

    # --- LSTM Forecast ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    window = 10
    last_window = scaled_data[-window:]
    cur_input = last_window.reshape(1, window, 1)
    forecast_lstm = []

    for _ in range(n_forecast):
        pred = lstm_model.predict(cur_input, verbose=0)[0][0]
        forecast_lstm.append(pred)
        cur_input = np.append(cur_input[:, 1:, :], [[[pred]]], axis=1)

    forecast_lstm = scaler.inverse_transform(np.array(forecast_lstm).reshape(-1, 1)).flatten()

    # --- Hybrid Forecast ---
    hybrid_forecast = (arima_forecast + forecast_lstm) / 2.0

    return data, hybrid_forecast


# === Fungsi Membuat Grafik dan Convert ke Base64 ===
def plot_to_base64(history, hybrid_forecast, periode):
    plt.figure(figsize=(10, 5))
    plt.plot(history, label="History", color="blue")
    plt.plot(
        range(len(history), len(history) + len(hybrid_forecast)),
        hybrid_forecast,
        color="green",
        linewidth=2,
        label=f"Hybrid Forecast ({periode} Bulan)"
    )
    plt.title(f"Prediksi Harga Saham Hybrid ARIMA–LSTM ({periode} Bulan)")
    plt.xlabel("Waktu")
    plt.ylabel("Harga Saham")
    plt.legend()
    plt.tight_layout()

    # Convert ke base64 supaya bisa ditampilkan di halaman HTML
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return img_base64


# === Deteksi Kolom Penutupan ===
def prepare_data(df):
    possible_cols = ["adj_close", "Adj Close", "close", "Close"]
    for c in possible_cols:
        if c in df.columns:
            df = df.rename(columns={c: "adj_close"})
            return df
    raise ValueError(f"Kolom harga penutupan tidak ditemukan. Kolom tersedia: {list(df.columns)}")


# === ROUTES ===
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    df = pd.read_csv(file)
    df = prepare_data(df)
    history = df["adj_close"].values

    results = []
    for bulan in [1, 2, 3]:
        n_forecast = bulan * 30
        _, hybrid_forecast = predict_stocks(df, n_forecast)
        img_base64 = plot_to_base64(history, hybrid_forecast, bulan)
        results.append({
            "bulan": bulan,
            "image": img_base64,
            "keterangan": f"Prediksi harga saham selama {bulan} bulan ke depan menggunakan model Hybrid ARIMA–LSTM."
        })

    return render_template("result.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
