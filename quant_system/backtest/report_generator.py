from quant_system.backtest.equity_curve import equity_curve_chart
from quant_system.backtest.drawdown_plot import drawdown_chart
from quant_system.backtest.heatmaps import session_heatmap, regime_heatmap, tier_heatmap
from quant_system.backtest.tier_breakdown import tier_breakdown
from quant_system.backtest.ml_explain import explain_trade
from quant_system.backtest.moonshots import moonshot_table
# EQUITY CURVE
equity_curve_chart(df)
st.markdown("<br>", unsafe_allow_html=True)

# DRAWDOWN
drawdown_chart(df)
st.markdown("<br>", unsafe_allow_html=True)

# HEATMAPS
session_heatmap(df)
regime_heatmap(df)
tier_heatmap(df)
st.markdown("<br>", unsafe_allow_html=True)

# TIER BREAKDOWN
tier_breakdown(df)
st.markdown("<br>", unsafe_allow_html=True)

# MOONSHOTS
moonshot_table(df)
st.markdown("<br>", unsafe_allow_html=True)


st.subheader("Explain a Trade")
selected_trade = st.selectbox("Select Trade ID", df.index.tolist())
row = df.loc[[selected_trade]]
features = row.filter(regex="feature_")  # your feature columns
explain_trade(model_version="latest", trade_features=features)
