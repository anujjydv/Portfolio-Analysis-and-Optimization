import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📊", layout="wide")

st.title("📊 Portfolio Analysis & Optimization")
st.markdown("CAPM stuff plus Markowitz portfolio optimization - efficient frontier, max Sharpe, min variance.")

st.sidebar.header("Settings")

tickers_input = st.sidebar.text_input("Stock tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

market_index = st.sidebar.text_input("Market benchmark ticker", value="^GSPC")

years_back = st.sidebar.slider("Years of historical data", 1, 10, 5)
end_date = date.today()
start_date = end_date - timedelta(days=years_back * 365)

rf_pct = st.sidebar.number_input("Risk-free rate (annual %)", min_value=0.0, max_value=20.0, value=4.5, step=0.1)
rf = rf_pct / 100

st.sidebar.markdown("---")
st.sidebar.caption("Beta > 1 means more volatile than the market, < 1 means less volatile.")

run_button = st.sidebar.button("Run Analysis", type="primary")


@st.cache_data(show_spinner=False)
def get_prices(symbols, start, end):
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = symbols
    return prices.dropna(how="all")


def daily_returns(prices):
    return prices.pct_change().dropna()


def get_beta(stock_rets, mkt_rets):
    aligned = pd.concat([stock_rets, mkt_rets], axis=1).dropna()
    aligned.columns = ["stock", "market"]
    cov = np.cov(aligned["stock"], aligned["market"])[0][1]
    mkt_var = np.var(aligned["market"])
    return cov / mkt_var


def annualize(daily_rets, trading_days=252):
    return (1 + daily_rets.mean()) ** trading_days - 1


def capm_return(beta, rf, mkt_return):
    return rf + beta * (mkt_return - rf)


def port_perf(weights, mean_returns, cov_matrix, trading_days=252):
    ret = np.sum(mean_returns * weights) * trading_days
    vol = np.sqrt(weights.T @ (cov_matrix * trading_days) @ weights)
    return ret, vol


def neg_sharpe(weights, mean_returns, cov_matrix, rf):
    ret, vol = port_perf(weights, mean_returns, cov_matrix)
    return -(ret - rf) / vol


def port_vol(weights, mean_returns, cov_matrix):
    return port_perf(weights, mean_returns, cov_matrix)[1]


def get_max_sharpe(mean_returns, cov_matrix, rf, bounds):
    n = len(mean_returns)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    guess = np.repeat(1 / n, n)
    return minimize(neg_sharpe, guess, args=(mean_returns, cov_matrix, rf), method="SLSQP",
                     bounds=bounds, constraints=cons)


def get_min_variance(mean_returns, cov_matrix, bounds):
    n = len(mean_returns)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    guess = np.repeat(1 / n, n)
    return minimize(port_vol, guess, args=(mean_returns, cov_matrix), method="SLSQP",
                     bounds=bounds, constraints=cons)


def get_efficient_frontier(mean_returns, cov_matrix, bounds, targets):
    n = len(mean_returns)
    vols = []
    for target in targets:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: port_perf(w, mean_returns, cov_matrix)[0] - target},
        )
        guess = np.repeat(1 / n, n)
        res = minimize(port_vol, guess, args=(mean_returns, cov_matrix), method="SLSQP",
                        bounds=bounds, constraints=cons)
        vols.append(res.fun if res.success else np.nan)
    return np.array(vols)


def simulate_portfolios(mean_returns, cov_matrix, rf, n=3000):
    n_assets = len(mean_returns)
    out = np.zeros((3, n))
    for i in range(n):
        w = np.random.random(n_assets)
        w /= w.sum()
        ret, vol = port_perf(w, mean_returns, cov_matrix)
        out[0, i] = vol
        out[1, i] = ret
        out[2, i] = (ret - rf) / vol
    return out


def port_daily_returns(weights, returns_df):
    return returns_df.dot(weights)


def sortino_ratio(rets, rf, trading_days=252):
    downside = rets[rets < 0]
    downside_std = downside.std() * np.sqrt(trading_days)
    ann_ret = annualize(rets, trading_days)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return (ann_ret - rf) / downside_std


def max_drawdown(rets):
    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return dd.min()


def var_95(rets):
    return -np.percentile(rets, 5)


def beta_alpha(port_rets, mkt_rets, rf, mkt_ann_return, port_ann_return):
    aligned = pd.concat([port_rets, mkt_rets], axis=1).dropna()
    aligned.columns = ["port", "mkt"]
    beta = np.cov(aligned["port"], aligned["mkt"])[0][1] / np.var(aligned["mkt"])
    alpha = port_ann_return - capm_return(beta, rf, mkt_ann_return)
    return beta, alpha


def treynor_ratio(port_ann_return, rf, beta):
    if beta == 0 or np.isnan(beta):
        return np.nan
    return (port_ann_return - rf) / beta


def risk_metrics(weights, returns_df, mkt_rets, rf, mkt_ann_return):
    rets = port_daily_returns(weights, returns_df)
    ann_ret, ann_vol = port_perf(weights, returns_df.mean(), returns_df.cov())
    sharpe = (ann_ret - rf) / ann_vol if ann_vol != 0 else np.nan
    sortino = sortino_ratio(rets, rf)
    beta, alpha = beta_alpha(rets, mkt_rets, rf, mkt_ann_return, ann_ret)
    treynor = treynor_ratio(ann_ret, rf, beta)
    return {
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Treynor Ratio": treynor,
        "Beta": beta,
        "Alpha": alpha,
        "Max Drawdown": max_drawdown(rets),
        "VaR 95%": var_95(rets),
    }


if run_button:
    st.session_state.analysis_run = True
    st.session_state.tickers = tickers
    st.session_state.market_index = market_index
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    st.session_state.rf = rf
    st.session_state.rf_pct = rf_pct

if not st.session_state.get("analysis_run", False):
    st.info("👈 Set your tickers and settings in the sidebar, then click **Run Analysis** to get started.")
    st.stop()

tickers = st.session_state.get("tickers", tickers)
market_index = st.session_state.get("market_index", market_index)
start_date = st.session_state.get("start_date", start_date)
end_date = st.session_state.get("end_date", end_date)
rf = st.session_state.get("rf", rf)
rf_pct = st.session_state.get("rf_pct", rf_pct)

if True:
    if not tickers:
        st.warning("Add at least one ticker.")
        st.stop()
    if len(tickers) < 2:
        st.info("Add 2+ stocks to unlock the portfolio optimizer.")

    with st.spinner("Fetching data..."):
        try:
            prices = get_prices(tickers + [market_index], start_date, end_date)
        except Exception as e:
            st.error(f"Couldn't download data: {e}")
            st.stop()

    if prices.empty or market_index not in prices.columns:
        st.error("No data came back - check your tickers.")
        st.stop()

    returns = daily_returns(prices)
    mkt_rets = returns[market_index]
    mkt_ann_return = annualize(mkt_rets)
    stock_rets = returns[[t for t in tickers if t in returns.columns]]

    tab_capm, tab_opt = st.tabs(["📈 CAPM", "🧮 Portfolio Optimizer"])

    with tab_capm:
        st.subheader("Normalized Prices")
        norm = prices / prices.iloc[0] * 100
        fig = px.line(norm, x=norm.index, y=norm.columns,
                       labels={"value": "Normalized Price (Base=100)", "x": "Date", "variable": "Ticker"})
        st.plotly_chart(fig, width="stretch")

        st.subheader("Beta & CAPM Expected Return")
        rows = []
        for t in tickers:
            if t not in returns.columns:
                continue
            beta = get_beta(returns[t], mkt_rets)
            exp_ret = capm_return(beta, rf, mkt_ann_return)
            actual_ret = annualize(returns[t])
            rows.append({
                "Ticker": t,
                "Beta": round(beta, 3),
                "CAPM Expected Return": f"{exp_ret * 100:.2f}%",
                "Actual Annual Return": f"{actual_ret * 100:.2f}%",
            })
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, width="stretch", hide_index=True)
        st.caption(f"Market ({market_index}) annual return: {mkt_ann_return * 100:.2f}% | Risk-free rate: {rf_pct:.2f}%")

        st.subheader("Beta Comparison")
        beta_fig = go.Figure()
        beta_fig.add_trace(go.Bar(x=results_df["Ticker"], y=[float(b) for b in results_df["Beta"]], marker_color="steelblue"))
        beta_fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Market Beta = 1")
        beta_fig.update_layout(yaxis_title="Beta", xaxis_title="Ticker")
        st.plotly_chart(beta_fig, width="stretch")

        st.subheader("Security Market Line")
        beta_range = np.linspace(0, max(2, results_df["Beta"].max() + 0.5), 50)
        sml_y = [capm_return(b, rf, mkt_ann_return) * 100 for b in beta_range]
        sml_fig = go.Figure()
        sml_fig.add_trace(go.Scatter(x=beta_range, y=sml_y, mode="lines", name="SML", line=dict(color="gray", dash="dash")))
        sml_fig.add_trace(go.Scatter(
            x=results_df["Beta"],
            y=[capm_return(b, rf, mkt_ann_return) * 100 for b in results_df["Beta"]],
            mode="markers+text", text=results_df["Ticker"], textposition="top center",
            marker=dict(size=12, color="crimson"), name="Stocks"
        ))
        sml_fig.update_layout(xaxis_title="Beta", yaxis_title="Expected Return (%)")
        st.plotly_chart(sml_fig, width="stretch")

        st.markdown("---")
        with st.expander("What is CAPM?"):
            st.markdown("""
CAPM estimates a stock's expected return based on how much systematic risk (beta) it carries:

**E(R) = Rf + β × (Rm − Rf)**

Beta of 1 = moves with the market. Above 1 = more volatile than the market, below 1 = less.
""")

    with tab_opt:
        if stock_rets.shape[1] < 2:
            st.warning("Need at least 2 valid tickers for optimization.")
        else:
            st.subheader("Markowitz Optimization")
            st.caption("Each dot is a randomly simulated portfolio. The line is the efficient frontier - best return for a given risk level.")

            mean_returns = stock_rets.mean()
            cov_matrix = stock_rets.cov()
            n_assets = len(mean_returns)
            bounds = tuple((0, 1) for _ in range(n_assets))

            allow_short = st.checkbox("Allow short-selling", value=False)
            if allow_short:
                bounds = tuple((-1, 1) for _ in range(n_assets))

            with st.spinner("Optimizing..."):
                max_sharpe_res = get_max_sharpe(mean_returns, cov_matrix, rf, bounds)
                min_var_res = get_min_variance(mean_returns, cov_matrix, bounds)

                ms_weights = max_sharpe_res.x
                mv_weights = min_var_res.x

                ms_ret, ms_vol = port_perf(ms_weights, mean_returns, cov_matrix)
                mv_ret, mv_vol = port_perf(mv_weights, mean_returns, cov_matrix)
                ms_sharpe = (ms_ret - rf) / ms_vol
                mv_sharpe = (mv_ret - rf) / mv_vol

                sim = simulate_portfolios(mean_returns, cov_matrix, rf)

                targets = np.linspace(mv_ret, mean_returns.max() * 252 * 0.98, 40)
                frontier_vols = get_efficient_frontier(mean_returns, cov_matrix, bounds, targets)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🏆 Max Sharpe Portfolio")
                st.metric("Expected Return", f"{ms_ret * 100:.2f}%")
                st.metric("Volatility", f"{ms_vol * 100:.2f}%")
                st.metric("Sharpe Ratio", f"{ms_sharpe:.3f}")
            with col2:
                st.markdown("### 🛡️ Min Variance Portfolio")
                st.metric("Expected Return", f"{mv_ret * 100:.2f}%")
                st.metric("Volatility", f"{mv_vol * 100:.2f}%")
                st.metric("Sharpe Ratio", f"{mv_sharpe:.3f}")

            st.subheader("Efficient Frontier")
            ef_fig = go.Figure()
            ef_fig.add_trace(go.Scatter(
                x=sim[0, :] * 100, y=sim[1, :] * 100, mode="markers",
                marker=dict(size=5, color=sim[2, :], colorscale="Viridis", colorbar=dict(title="Sharpe"), showscale=True, opacity=0.5),
                name="Simulated Portfolios", hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
            ))
            valid = ~np.isnan(frontier_vols)
            ef_fig.add_trace(go.Scatter(x=frontier_vols[valid] * 100, y=targets[valid] * 100, mode="lines",
                                         name="Efficient Frontier", line=dict(color="black", width=3)))
            ef_fig.add_trace(go.Scatter(x=[ms_vol * 100], y=[ms_ret * 100], mode="markers", name="Max Sharpe",
                                         marker=dict(color="red", size=16, symbol="star")))
            ef_fig.add_trace(go.Scatter(x=[mv_vol * 100], y=[mv_ret * 100], mode="markers", name="Min Variance",
                                         marker=dict(color="blue", size=14, symbol="diamond")))
            cal_x = np.linspace(0, ms_vol * 1.3, 20)
            cal_y = rf + (ms_ret - rf) / ms_vol * cal_x
            ef_fig.add_trace(go.Scatter(x=cal_x * 100, y=cal_y * 100, mode="lines", name="Capital Allocation Line",
                                         line=dict(color="orange", dash="dot")))
            ef_fig.update_layout(xaxis_title="Volatility (%)", yaxis_title="Expected Return (%)",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02), height=550)
            st.plotly_chart(ef_fig, width="stretch")

            st.subheader("Optimal Weights")
            weights_df = pd.DataFrame({
                "Ticker": mean_returns.index,
                "Max Sharpe": (ms_weights * 100).round(2),
                "Min Variance": (mv_weights * 100).round(2),
            })
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                st.markdown("**Max Sharpe Allocation**")
                st.plotly_chart(px.pie(weights_df, names="Ticker", values="Max Sharpe", hole=0.4), width="stretch")
            with wcol2:
                st.markdown("**Min Variance Allocation**")
                st.plotly_chart(px.pie(weights_df, names="Ticker", values="Min Variance", hole=0.4), width="stretch")
            st.dataframe(weights_df, width="stretch", hide_index=True)

            st.subheader("📐 Risk Metrics")
            st.caption("Sharpe, Sortino, Treynor, Beta, Alpha, Max Drawdown and VaR for each portfolio, plus an equal-weight benchmark.")

            equal_weights = np.repeat(1 / n_assets, n_assets)
            portfolios = {"Max Sharpe": ms_weights, "Min Variance": mv_weights, "Equal Weight": equal_weights}

            metrics = {name: risk_metrics(w, stock_rets, mkt_rets, rf, mkt_ann_return) for name, w in portfolios.items()}
            metrics_df = pd.DataFrame(metrics).T

            display_df = pd.DataFrame(index=metrics_df.index)
            display_df["Annual Return"] = (metrics_df["Annual Return"] * 100).round(2).astype(str) + "%"
            display_df["Annual Volatility"] = (metrics_df["Annual Volatility"] * 100).round(2).astype(str) + "%"
            display_df["Sharpe"] = metrics_df["Sharpe Ratio"].round(3)
            display_df["Sortino"] = metrics_df["Sortino Ratio"].round(3)
            display_df["Treynor"] = metrics_df["Treynor Ratio"].round(3)
            display_df["Beta"] = metrics_df["Beta"].round(3)
            display_df["Alpha"] = (metrics_df["Alpha"] * 100).round(2).astype(str) + "%"
            display_df["Max Drawdown"] = (metrics_df["Max Drawdown"] * 100).round(2).astype(str) + "%"
            display_df["VaR 95% (1-day)"] = (metrics_df["VaR 95%"] * 100).round(2).astype(str) + "%"
            st.dataframe(display_df, width="stretch")

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            best_sharpe = metrics_df["Sharpe Ratio"].idxmax()
            mcol1.metric("Best Sharpe", best_sharpe, f"{metrics_df.loc[best_sharpe, 'Sharpe Ratio']:.3f}")
            best_sortino = metrics_df["Sortino Ratio"].idxmax()
            mcol2.metric("Best Sortino", best_sortino, f"{metrics_df.loc[best_sortino, 'Sortino Ratio']:.3f}")
            shallow_dd = metrics_df["Max Drawdown"].idxmax()
            mcol3.metric("Shallowest Drawdown", shallow_dd, f"{metrics_df.loc[shallow_dd, 'Max Drawdown'] * 100:.2f}%")
            low_var = metrics_df["VaR 95%"].idxmin()
            mcol4.metric("Lowest VaR", low_var, f"{metrics_df.loc[low_var, 'VaR 95%'] * 100:.2f}%")

            with st.expander("What do these numbers mean?"):
                st.markdown("""
- **Sharpe** - return per unit of total risk. Higher is better.
- **Sortino** - like Sharpe but only counts downside volatility. Upside swings don't hurt this one.
- **Treynor** - return per unit of market risk (beta) instead of total volatility.
- **Beta** - how much the portfolio moves with the market.
- **Alpha** - return above what CAPM would predict given the beta. Positive means it's beating the market on a risk-adjusted basis.
- **Max Drawdown** - worst peak-to-trough loss over the period.
- **VaR 95% (1-day)** - the daily loss you shouldn't expect to exceed 95% of the time, based on history.
""")

            st.markdown("---")
            with st.expander("What is Markowitz optimization?"):
                st.markdown("""
Modern Portfolio Theory finds the mix of assets that gives the best return for a given level of risk.

- **Efficient Frontier** - portfolios that give the highest return for each risk level. Anything below it is leaving return on the table.
- **Max Sharpe Portfolio** - best risk-adjusted return on the frontier, also called the tangency portfolio.
- **Min Variance Portfolio** - lowest possible risk, regardless of return.
- **Capital Allocation Line** - the line from the risk-free rate through the Max Sharpe portfolio.

Weights default to long-only (0-100%). Check "Allow short-selling" to let weights go negative.
""")