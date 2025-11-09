# sip_yfinace/sip_yfinace_ai.py
import io, base64
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from dateutil.relativedelta import relativedelta


class SIPYFinanceAI:
    """
    Full AI-powered SIP calculator and predictor (Fixed version)
    """

    def __init__(self, ticker="NIFTYBEES.NS", years=10, horizon_months=12,
                 sip_amount=1000, expected_return=12, duration_years=10):
        self.ticker = ticker.upper()
        self.years = years
        self.horizon = horizon_months
        self.sip_amount = sip_amount
        self.expected_return = expected_return
        self.duration_years = duration_years
        self.df = None
        self.model = None

    # Step 1: Fetch 10 years of price data
    def fetch_data(self):
        end = pd.Timestamp.today()
        start = end - relativedelta(years=self.years)
        df = yf.download(self.ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data found for {self.ticker}")
        df = df[["Close"]].rename(columns={"Close": "close"})
        # Fix warning: use 'ME' instead of 'M'
        df = df.resample("ME").last().dropna()
        df["ret_1m"] = df["close"].pct_change()
        df.dropna(inplace=True)
        self.df = df
        return df

    # Step 2: Create features for XGBoost
    def make_features(self, df):
        feat = df.copy()
        for lag in [1, 2, 3, 6, 9, 12]:
            feat[f"lag_{lag}"] = feat["ret_1m"].shift(lag)
        feat["roll_mean_3"] = feat["ret_1m"].rolling(3).mean()
        feat["roll_std_3"] = feat["ret_1m"].rolling(3).std()
        feat["target"] = feat["ret_1m"].shift(-1)
        feat.dropna(inplace=True)
        return feat

    # Step 3: Train and evaluate XGBoost
    def train_model(self, feat):
        X = feat.drop(columns=["target"])
        y = feat["target"]
        tscv = TimeSeriesSplit(n_splits=5)
        for tr, te in tscv.split(X):
            train, test = tr, te

        model = XGBRegressor(n_estimators=400, learning_rate=0.05,
                             max_depth=4, subsample=0.9, colsample_bytree=0.9)
        model.fit(X.iloc[train], y.iloc[train])
        y_pred = model.predict(X.iloc[test])
        mae = mean_absolute_error(y.iloc[test], y_pred)
        self.model = model

        test_df = pd.DataFrame({
            "Date": y.iloc[test].index,
            "Actual": y.iloc[test].values,
            "Predicted": y_pred
        })
        test_df["Difference"] = test_df["Actual"] - test_df["Predicted"]
        return test_df, mae

    # Step 4: Predict next N months using last known values
    def forecast(self, feat):
        last_row = feat.iloc[-1].copy()
        forecasts = []
        lag_cols = [1, 2, 3, 6, 9, 12]

        for _ in range(self.horizon):
            x_input = last_row.drop(labels=["target"]).values.reshape(1, -1)
            next_pred = float(self.model.predict(x_input)[0])
            forecasts.append(next_pred)

            # Shift only existing lag columns safely
            last_row["ret_1m"] = next_pred
            for k in reversed(lag_cols):
                if k == 1:
                    last_row["lag_1"] = next_pred
                else:
                    prev = [p for p in lag_cols if p < k]
                    if prev:
                        last_row[f"lag_{k}"] = last_row[f"lag_{prev[-1]}"]
            # Rolling stats update (approximate)
            last_row["roll_mean_3"] = (last_row["roll_mean_3"] + next_pred) / 2
            last_row["roll_std_3"] = abs(last_row["roll_std_3"] * 0.9 + abs(next_pred) * 0.1)
        return forecasts

    # Step 5: Calculate SIP result
    def sip_calculator(self):
        r = self.expected_return / 12 / 100
        n = self.duration_years * 12
        future_value = self.sip_amount * (((1 + r) ** n - 1) / r) * (1 + r)
        invested = self.sip_amount * n
        gain = future_value - invested
        return {
            "invested": invested,
            "future_value": future_value,
            "gain": gain,
        }

    # Step 6: Plot comparison graph
    def plot_graphs(self, df_res, sip_result, forecasts):
        # 1. Actual vs Predicted
        plt.figure(figsize=(10, 5))
        plt.plot(df_res["Date"], df_res["Actual"], label="Actual Return")
        plt.plot(df_res["Date"], df_res["Predicted"], label="Predicted Return")
        plt.title("Actual vs Predicted Returns (Test)")
        plt.legend()
        buf1 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf1, format="png")
        plt.close()
        img1 = base64.b64encode(buf1.getvalue()).decode()

        # 2. Forecast chart
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(forecasts)), forecasts, label="Forecasted Future Returns")
        plt.axhline(0, color="black", linestyle="--")
        plt.title(f"Next {self.horizon} Months Forecast")
        plt.legend()
        buf2 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf2, format="png")
        plt.close()
        img2 = base64.b64encode(buf2.getvalue()).decode()

        # 3. SIP growth comparison chart
        months = np.arange(1, self.duration_years * 12 + 1)
        r = self.expected_return / 12 / 100
        sip_balance = [self.sip_amount * (((1 + r) ** m - 1) / r) * (1 + r) for m in months]
        plt.figure(figsize=(10, 5))
        plt.plot(months, sip_balance, label="SIP Expected Growth")
        plt.title("SIP Investment Growth Over Time")
        plt.xlabel("Months")
        plt.ylabel("Portfolio Value")
        plt.legend()
        buf3 = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf3, format="png")
        plt.close()
        img3 = base64.b64encode(buf3.getvalue()).decode()

        return {"actual_pred": img1, "forecast": img2, "sip_growth": img3}

    # Step 7: AI Insights & Recommendations
    def ai_insights(self, mae, forecasts):
        avg_pred = np.mean(forecasts)
        vol_pred = np.std(forecasts)
        direction = "UP" if avg_pred > 0 else "DOWN"
        risk = "High" if vol_pred > 0.06 else ("Medium" if vol_pred > 0.03 else "Low")
        if avg_pred > 0 and risk in ["Low", "Medium"]:
            advice = "Market outlook positive. Continue SIPs or invest more gradually."
        elif avg_pred > 0 and risk == "High":
            advice = "Volatile phase ahead. Continue SIPs, avoid lump-sum."
        else:
            advice = "Caution advised. Stick to SIP; avoid large investments."
        return {
            "avg_predicted_return": avg_pred,
            "volatility": vol_pred,
            "direction": direction,
            "risk": risk,
            "recommendation": advice,
            "mae": mae,
        }

    # Step 8: Full pipeline
    def run(self):
        df = self.fetch_data()
        feat = self.make_features(df)
        df_res, mae = self.train_model(feat)
        forecasts = self.forecast(feat)
        sip_result = self.sip_calculator()
        graphs = self.plot_graphs(df_res, sip_result, forecasts)
        insights = self.ai_insights(mae, forecasts)
        return {
            "ticker": self.ticker,
            "mae": mae,
            "sip_result": sip_result,
            "forecast_list": forecasts,
            "insights": insights,
            "graphs": graphs,
            "diff_table": df_res.tail(20).to_dict(orient="records"),
        }


if __name__ == "__main__":
    ai = SIPYFinanceAI(
        ticker="NIFTYBEES.NS",
        years=10,
        horizon_months=6,
        sip_amount=2000,
        expected_return=12,
        duration_years=10
    )

    result = ai.run()

    print("=== SIP YFINANCE AI REPORT ===")
    print("Ticker:", result["ticker"])
    print("Model MAE:", result["mae"])
    print("Predicted Avg Return:", result["insights"]["avg_predicted_return"])
    print("Volatility:", result["insights"]["volatility"])
    print("Direction:", result["insights"]["direction"])
    print("Risk:", result["insights"]["risk"])
    print("AI Recommendation:", result["insights"]["recommendation"])
    print("SIP Future Value:", result["sip_result"]["future_value"])
    print("SIP Gain:", result["sip_result"]["gain"])
    print("✅ Graphs ready (base64 encoded)")
