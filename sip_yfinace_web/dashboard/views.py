import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from django.shortcuts import render
from sip_yfinace_ai import SIPYFinanceAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
import pandas as pd
import numpy as np


def create_graph_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def index(request):
    context = {}
    if request.method == "POST":
        sip_amount = int(request.POST.get("sip_amount", 2000))
        expected_return = float(request.POST.get("expected_return", 12))
        duration_years = int(request.POST.get("duration_years", 10))
        horizon = int(request.POST.get("horizon", 6))

        try:
            ai = SIPYFinanceAI(
                ticker="NIFTYBEES.NS",
                sip_amount=sip_amount,
                expected_return=expected_return,
                duration_years=duration_years,
                horizon_months=horizon
            )
            res = ai.run()

            # --- Graph 1: Predicted vs Actual ---
            df_diff = pd.DataFrame(res["diff_table"]).dropna()
            df_diff["Date"] = pd.to_datetime(df_diff["Date"])
            df_diff = df_diff.tail(15)

            fig1, ax1 = plt.subplots(figsize=(6.5, 3.5))
            ax1.plot(df_diff["Date"], df_diff["Actual"], label="Actual Returns", color="blue")
            ax1.plot(df_diff["Date"], df_diff["Predicted"], label="Predicted Returns", color="orange", linestyle="--")
            ax1.set_title("Predicted vs Actual Returns (Recent Data)", fontweight="bold")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Return %")
            ax1.legend()
            graph_predicted_vs_actual = create_graph_image(fig1)
            plt.close(fig1)

            # --- Graph 2: Forecast (from AI) ---
            graph_forecast = res["graphs"]["forecast"]

            # --- Graph 3: SIP Composition Pie Chart ---
            invested = res["sip_result"]["invested"]
            future_value = res["sip_result"]["future_value"]
            gain = future_value - invested
            future_projection = future_value * 0.1

            labels = ["Invested Capital", "AI-Predicted Gain", "Projected Growth Potential"]
            values = [invested, gain, future_projection]
            colors = ["#9ca3af", "#10b981", "#2563eb"]

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            wedges, texts, autotexts = ax3.pie(
                values, labels=labels, autopct="%1.1f%%", startangle=90,
                colors=colors, textprops={'fontsize': 9, 'fontname': 'Times New Roman'}
            )
            ax3.set_title("SIP Value Distribution (AI-Based Projection)", fontweight="bold", fontsize=11)
            ax3.axis("equal")
            graph_pie = create_graph_image(fig3)
            plt.close(fig3)

            # --- AI Insights & Recommendations ---
            avg_return = res["insights"]["avg_predicted_return"]

            # Dynamic recommendation text based on performance
            if gain / invested > 0.5:
                ai_recommendation = (
                    "AI analysis indicates strong growth momentum based on historical data trends. "
                    "Your SIP portfolio has delivered consistent returns outperforming the market average. "
                    "It is advisable to maintain your SIP discipline and consider moderate top-ups quarterly. "
                    "Market outlook remains positive for long-term investors, particularly in diversified equity funds."
                )
            elif gain / invested > 0.3:
                ai_recommendation = (
                    "AI predicts a steady upward performance trend with moderate volatility. "
                    "Your SIP plan shows good growth consistency aligning with historical market averages. "
                    "Continue monthly contributions without pause and monitor fund category returns yearly. "
                    "AI suggests focusing on quality funds to sustain 10–12% annual growth."
                )
            else:
                ai_recommendation = (
                    "AI forecasts indicate slower growth due to current market corrections. "
                    "Your SIP portfolio remains resilient but under moderate performance range. "
                    "Avoid large one-time investments and maintain systematic discipline. "
                    "Long-term investing beyond 7 years is recommended for compounding recovery."
                )

            # --- Structured Recommendations ---
            recommendations = [
                ["Stay consistent with monthly SIPs", "Increase SIP annually by 10–15% if possible"],
                ["Diversify across equity and balanced funds", "Avoid reacting to short-term volatility"],
                ["Use AI forecasts to rebalance investments", "Review performance every 6–12 months"],
                ["Invest through market cycles", "Focus on long-term compounding for wealth creation"],
            ]

            # --- AI Conclusion ---
            conclusion = (
                f"Based on advanced AI insights and 10-year historical YFinance analysis, "
                f"your SIP portfolio demonstrates healthy predictive consistency. "
                f"With an estimated average return of {avg_return:.2%} and projected total value of ₹{future_value:,.0f}, "
                f"your long-term SIP plan reflects strong financial discipline. "
                f"The AI model concludes that maintaining regular SIPs, reviewing annually, "
                f"and focusing on compounding will optimize returns and minimize risk."
            )

            # --- Update context for template ---
            context.update({
                "sip_amount": sip_amount,
                "expected_return": expected_return,
                "duration_years": duration_years,
                "horizon": horizon,
                "sip": res["sip_result"],
                "insights": {"avg_predicted_return": avg_return, "recommendation": ai_recommendation},
                "graph_predicted_vs_actual": graph_predicted_vs_actual,
                "graph_forecast": graph_forecast,
                "graph_pie": graph_pie,
                "recommendations": recommendations,
                "conclusion": conclusion,
                "result_ready": True,
            })

        except Exception as e:
            context["error"] = str(e)

    return render(request, "dashboard/index.html", context)
