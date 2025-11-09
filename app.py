from flask import Flask, render_template_string, request
from sip_yfinace_ai import SIPYFinanceAI
import datetime

app = Flask(__name__)

# HTML template is embedded for simplicity
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI SIP Calculator</title>
    <style>
        body { font-family: "Times New Roman", serif; background:#f6f7fb; margin:0; padding:0; }
        header { background:#111827; color:white; padding:12px 20px; }
        h1 { margin:0; }
        form { background:white; padding:20px; margin:20px auto; width:500px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.1); }
        label { display:block; margin-top:10px; }
        input { width:100%; padding:8px; margin-top:5px; border:1px solid #ccc; border-radius:6px; }
        button { margin-top:15px; padding:10px 15px; background:#2563eb; color:white; border:none; border-radius:6px; cursor:pointer; }
        button:hover { opacity:0.9; }
        .container { width:95%; max-width:1200px; margin:20px auto; }
        .card { background:white; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.05); padding:20px; margin-bottom:20px; }
        img { width:100%; border-radius:8px; margin-top:10px; }
        table { width:100%; border-collapse: collapse; margin-top:10px; }
        th, td { padding:8px; border-bottom:1px solid #eee; text-align:left; }
        .metrics { display:grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap:20px; }
        footer { text-align:center; padding:10px; color:#555; font-size:13px; }
    </style>
</head>
<body>
    <header><h1>AI-Powered SIP Calculator</h1></header>
    <div class="container">

        <form method="post">
            <label>Ticker (Yahoo Finance)</label>
            <input name="ticker" placeholder="e.g. NIFTYBEES.NS" value="{{ticker}}" required>
            
            <label>SIP Amount (₹)</label>
            <input name="sip_amount" type="number" value="{{sip_amount}}" required>
            
            <label>Expected Annual Return (%)</label>
            <input name="expected_return" type="number" step="0.1" value="{{expected_return}}" required>
            
            <label>Duration (Years)</label>
            <input name="duration_years" type="number" value="{{duration_years}}" required>
            
            <label>Forecast Horizon (Months)</label>
            <input name="horizon" type="number" value="{{horizon}}" required>
            
            <button type="submit">Run Analysis</button>
        </form>

        {% if result_ready %}
        <div class="card">
            <h2>AI Insights</h2>
            <div class="metrics">
                <div><b>Model MAE:</b> {{ insights.mae|round(5) }}</div>
                <div><b>Avg Predicted Return:</b> {{ insights.avg_predicted_return|round(5) }}</div>
                <div><b>Volatility:</b> {{ insights.volatility|round(5) }}</div>
                <div><b>Trend Direction:</b> {{ insights.direction }}</div>
                <div><b>Risk Band:</b> {{ insights.risk }}</div>
            </div>
            <p><b>AI Recommendation:</b><br>{{ insights.recommendation }}</p>
        </div>

        <div class="card">
            <h2>SIP Results</h2>
            <p><b>Invested:</b> ₹{{ sip_result.invested|round(2) }}</p>
            <p><b>Future Value:</b> ₹{{ sip_result.future_value|round(2) }}</p>
            <p><b>Gain:</b> ₹{{ sip_result.gain|round(2) }}</p>
        </div>

        <div class="card">
            <h2>Actual vs Predicted Returns (Test Data)</h2>
            <img src="data:image/png;base64,{{ graphs.actual_pred }}" alt="Actual vs Predicted">
        </div>

        <div class="card">
            <h2>Forecast (Next {{ horizon }} Months)</h2>
            <img src="data:image/png;base64,{{ graphs.forecast }}" alt="Forecast">
        </div>

        <div class="card">
            <h2>SIP Growth Over Time</h2>
            <img src="data:image/png;base64,{{ graphs.sip_growth }}" alt="SIP Growth">
        </div>

        <div class="card">
            <h2>Actual vs Predicted Table (last 20 test months)</h2>
            <table>
                <thead>
                    <tr><th>Date</th><th>Actual</th><th>Predicted</th><th>Diff</th></tr>
                </thead>
                <tbody>
                {% for row in diff_table %}
                    <tr>
                        <td>{{ row.Date|date:"Y-m-d" if row.Date else "" }}</td>
                        <td>{{ row.Actual|round(5) }}</td>
                        <td>{{ row.Predicted|round(5) }}</td>
                        <td>{{ row.Difference|round(5) }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
    </div>

    <footer>© {{year}} SIP YFinance AI | Powered by Flask + XGBoost</footer>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "ticker": "NIFTYBEES.NS",
        "sip_amount": 2000,
        "expected_return": 12,
        "duration_years": 10,
        "horizon": 6,
        "year": datetime.datetime.now().year,
        "result_ready": False
    }

    if request.method == "POST":
        try:
            ticker = request.form["ticker"]
            sip_amount = int(request.form["sip_amount"])
            expected_return = float(request.form["expected_return"])
            duration_years = int(request.form["duration_years"])
            horizon = int(request.form["horizon"])

            ai = SIPYFinanceAI(
                ticker=ticker,
                years=10,
                horizon_months=horizon,
                sip_amount=sip_amount,
                expected_return=expected_return,
                duration_years=duration_years
            )

            result = ai.run()

            context.update({
                "result_ready": True,
                "ticker": ticker,
                "sip_amount": sip_amount,
                "expected_return": expected_return,
                "duration_years": duration_years,
                "horizon": horizon,
                "insights": result["insights"],
                "sip_result": result["sip_result"],
                "graphs": result["graphs"],
                "diff_table": result["diff_table"]
            })

        except Exception as e:
            context["error"] = str(e)

    return render_template_string(TEMPLATE, **context)


if __name__ == "__main__":
    app.run(debug=True)
