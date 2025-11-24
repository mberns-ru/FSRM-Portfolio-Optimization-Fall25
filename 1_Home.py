import textwrap

import streamlit as st

# Optional: pull config from your training script if it's importable
try:
    import _gradientboost as gb

    TICKERS = gb.TICKERS
    START = gb.START
    END = gb.END
except Exception:
    TICKERS = None
    START = "2010-01-01"
    END = "2025-12-31"

st.set_page_config(
    page_title="Gradient Boosted Portfolio Lab",
    layout="wide",
)

st.title("🔮 ML Portfolio Lab")

st.markdown(
    """
Welcome to your **ML Portfolio Lab**. This app lets you run and compare three
Masuda-style portfolio models:

- 📈 **Gradient Boost (XGBoost / GBM)**  
- 🌲 **Random Forest**  
- 🧬 **PCA + LightGBM**  

Each model trains on historical prices, builds a monthly ML-optimized portfolio,
and compares its performance to an S&P 500 benchmark.
"""
)

# ------------------------
# 1. High-level workflow
# ------------------------
st.markdown("## 🚦 Workflow Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ Run backtests")
    st.markdown(
        """
        Go to the **“🧪 Run & Save ML Portfolio Backtests”** page to:

        - Choose a model (GBM, RF, PCA+LGBM)  
        - Set **start / end dates**  
        - Adjust **finance parameters**:  
          - Risk aversion (λ)  
          - Transaction costs (bps)  
          - Risk-free rate (for Sharpe)  
        - Click **Run backtest** to create a timestamped `.pkl` in `results/`.  
        """
    )

with col2:
    st.markdown("### 2️⃣ Inspect model results")
    st.markdown(
        """
        Each model has its own results page:

        - 📈 **Gradient Boost Portfolio – Backtest Results**  
        - 🌲 **Random Forest Portfolio – Backtest Results**  
        - 🧬 **PCA + LightGBM Portfolio – Results**  

        On each page you can:

        - Plot 2025 **equity curves** (ML vs SPY)  
        - See **bi-monthly portfolio weights** in 2025  
        - View a **Trades / Buys / Sells / Holds** table (2024–2025)  
        - Examine **2-month returns (2024–2025)**  
        - Compare **train vs test metrics**  
        """
    )

with col3:
    st.markdown("### 3️⃣ Compare models side-by-side")
    st.markdown(
        """
        Use the **“⚖️ Model Comparison – GBM vs RF vs PCA+LGBM”** page to:

        - Contrast **bi-monthly ML_Opt returns** across models  
        - Overlay **2025 equity curves** for all models + SPY  
        - Compare **test-period performance metrics** (Sharpe, drawdown, etc.)  

        This is your summary view for deciding which model behaves best under
        your current finance assumptions.
        """
    )

# ------------------------
# 2. Current configuration
# ------------------------
st.markdown("## ⚙️ Current Data & Universe (from training scripts)")

conf_left, conf_right = st.columns(2)

with conf_left:
    st.markdown("### Data & Universe")

    if TICKERS is not None:
        st.markdown(
            f"- **Sample window**: `{START}` → `{END}`\n"
            f"- **Universe size**: `{len(TICKERS)}` tickers\n"
        )
        st.markdown("**Tickers:**")
        st.code(", ".join(TICKERS), language="text")
    else:
        st.warning(
            "Could not import `_gradientboost`. "
            "If you renamed the training file, update the import at the top of this page."
        )
        st.markdown(
            f"- **Sample window**: `{START}` → `{END}` (default placeholder)\n"
            "- **Tickers**: unknown (set in your training scripts)\n"
        )

with conf_right:
    st.markdown("### What each `.pkl` contains")

    st.markdown(
        """
        Each run of any model script (GBM / RF / PCA+LGBM) saves a results dictionary
        with at least:

        - **`monthly`**: monthly strategy & benchmark returns  
        - **`weights`**: portfolio weights at each rebalance  
        - **`monthly_train` / `monthly_test`**: 2010–2023 vs 2024–2025 split  
        - **`weights_2025_bimonth`**: bi-monthly weights in 2025  
        - **`equity_2025`**: $1000 ML portfolio vs $1000 SPY (2025)  
        - **`metrics`**: train / test performance metrics (Sharpe, drawdown, etc.)  
        """
    )

# ------------------------
# 3. Usage notes
# ------------------------
st.markdown("## 📝 Usage Notes")

st.markdown(
    textwrap.dedent(
        """
        - **Finance parameters live on the Run Models page**  
          You control the **economic assumptions** there: risk aversion λ, transaction
          costs in basis points, and the risk-free rate used for Sharpe. The ML model
          hyperparameters stay fixed inside the scripts.

        - **No accidental retraining**  
          The results pages and comparison page only **load and visualize** saved
          `.pkl` files. A new backtest only runs when you click **Run backtest** on
          the dedicated page.

        - **Versioning by timestamp**  
          Each run writes a new file like:
          `gradientboost_results_YYYYMMDD_HHMMSS.pkl`,
          `randomforest_results_YYYYMMDD_HHMMSS.pkl`,
          or `pca_lightgbm_results_YYYYMMDD_HHMMSS.pkl`.  
          Keep multiple runs side-by-side to compare different universes or
          finance parameters.
        """
    )
)

# ------------------------
# 4. Navigation hints
# ------------------------
st.markdown("## 🧭 Where to go next")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown("### ➡️ Run and save new backtests")
    st.markdown(
        """
        - Open the **“🧪 Run & Save ML Portfolio Backtests”** page  
        - Pick a model and set finance parameters  
        - Run a backtest and download the `.pkl` if you want  
        """
    )

with nav_col2:
    st.markdown("### 📊 Explore results & comparisons")
    st.markdown(
        """
        - Use the three model-specific pages (📈 / 🌲 / 🧬) to dive into a single model  
        - Use **“⚖️ Model Comparison – GBM vs RF vs PCA+LGBM”** to see them together  
        """
    )
