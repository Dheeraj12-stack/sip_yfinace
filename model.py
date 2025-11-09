# sip_yfinace/model.py
import io, base64
import pandas as pd, numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SIPYFinanceAI:
    """
    Stand-alone class for SIP analysis using yfinance + XGBoost
    """

    def __init__(self, ticker="NIFTYBEES.NS", years=10, horizon_months=6):
        self.ticker = ticker.upper()
        self.years = years
        self.horizon = horizon_months
        self.data = None
        self.model = None

    # --- Step 1: Fetch data ---
    def fetch_data(self):
        end = pd.Timestamp.today()
        start = end - relativedelta(years=self.years)
        df = yf.download(self.ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data found for {self.ticker}")
        df = df[["Close"]].rename(columns={"Close": "close"})
        df = df.resample("M").last().dropna()
        df["ret_1m"] = df["close"].pct_change()
        df.dropna(inplace=True)
        self.data = df
        return df

    # --- Step 2: Create features ---
    def make_features(self, df):
        feat = df.copy()
        for lag in [1, 2, 3, 6, 9, 12]:
            feat[f"lag_{lag}"] = feat["ret_1m"].shift(lag)
        feat["roll_mean_3"] = feat["ret_1m"].rolling(3).mean()
        feat["roll_std_3"] = feat["ret_1m"].rolling(3).std()
        feat["target"] = feat["ret_1m"].shift(-1)
        feat.dropna(inplace=True)
        return feat

    # --- Step 3: Train model ---
    def train_model(self, feat):
        X = feat.drop(columns=["target"])
        y = feat["target"]
        tscv = TimeSeriesSplit(n_splits=5)
        for tr, te in tscv.split(X):
            train, test = tr, te

        model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4)
        model.fit(X.iloc[train], y.iloc[train])
        y_pred = model.predict(X.iloc[test])
        mae = mean_absolute_error(y.iloc[test], y_pred)
        self.model = model

        # Prepare test results
        result_df = pd.DataFrame({
            "Date": y.iloc[test].index,
            "Actual": y.iloc[test].values,
            "Predicted": y_pred
        })
        result_df["Difference"] = result_df["Actual"] - result_df["Predicted"]

        return result_df, mae

    # --- Step 4: Plot actual vs predicted ---
    def plot_results(self, df_res):
        plt.figure(figsize=(8, 4))
        plt.plot(df_res["Date"], df_res["Actual"], label="Actual")
        plt.plot(df_res["Date"], df_res["Predicted"], label="Predicted")
        plt.legend()
        plt.title(f"Actual vs Predicted Returns — {self.ticker}")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # --- Step 5: Run the full pipeline ---
    def run(self):
        df = self.fetch_data()
        feat = self.make_features(df)
        df_res, mae = self.train_model(feat)
        img_b64 = self.plot_results(df_res)

        insights = {
            "mae": float(mae),
            "latest_actual": float(df_res["Actual"].iloc[-1]),
            "latest_pred": float(df_res["Predicted"].iloc[-1]),
            "avg_forecast": float(df_res["Predicted"].mean()),
            "volatility": float(df_res["Predicted"].std()),
        }

        return {
            "ticker": self.ticker,
            "metrics": insights,
            "chart": img_b64,
            "diff_table": df_res.tail(20).to_dict(orient="records"),
        }


if __name__ == "__main__":
    ai = SIPYFinanceAI("NIFTYBEES.NS", years=10, horizon_months=6)
    result = ai.run()

    print("Model MAE:", result["metrics"]["mae"])
    print("Average Predicted Return:", result["metrics"]["avg_forecast"])
    print("Volatility:", result["metrics"]["volatility"])
    print("✅ Chart ready as Base64 string (use in HTML if needed)")
