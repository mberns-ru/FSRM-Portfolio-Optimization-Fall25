import textwrap
import streamlit as st

# Optional: pull config from your GBM training script if it's importable
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
    page_title="ML Portfolio Lab",
    layout="wide",
)

st.title("🔮 ML Portfolio Lab")

st.markdown(
    """
Welcome to your **ML Portfolio Lab**.  
This app now supports **four full Masuda-style ML portfolio models**:

- 📈 **Gradient Boost (XGBoost / GBM)**  
- 🌲 **Random Forest**  
- 🧬 **PCA + LightGBM**  
- 🐈 **CatBoost**  

Each model trains on historical prices, constructs monthly ML-optimized portfolios,
and compares performance to the **S&P 500 benchmark**.
"""
)

# ------------------------
# 1. High-level workflow
# ------------------------
st.markdown("## 🚦 Workflow Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 1️⃣ Run backtests")
    st.markdown(
        """
        Visit **“🧪 Run & Save ML Portfolio Backtests”** to:

        - Choose a model:  
          - GBM  
          - Random Forest  
          - PCA + LightGBM  
          - CatBoost  
        - Set **start / end dates**  
        - Adjust **finance parameters** (λ, transaction costs, risk-free rate)  
        - Run and save `.pkl` result files  
        """
    )

with col2:
    st.markdown("### 2️⃣ Inspect results per model")
    st.markdown(
        """
        Each model has its own dedicated results page:

        - 📈 **Gradient Boost – Backtest Results**  
        - 🌲 **Random Forest – Backtest Results**  
        - 🧬 **PCA + LightGBM – Results**  
        - 🐈 **CatBoost – Backtest Results**  

        Explore:

        - 2025 **equity curves** (ML vs SPY)  
        - **Bi-monthly weights** in 2025  
        - **Trades / Buys / Sells / Holds** (2024–2025)  
        - **2-month returns** (2024–2025)  
        - **Train vs Test metrics**  
        """
    )

with col3:
    st.markdown("### 3️⃣ Compare all 4 models")
    st.markdown(
        """
        Use **“⚖️ Model Comparison – GBM vs RF vs PCA+LGBM vs CatBoost”** to:

        - Overlay **2025 equity curves** for all 4 models + SPY  
        - Compare **test-period Sharpe, vol, drawdown**  
        - View **side-by-side ML_Opt returns**  
        - Evaluate stability of each model’s ML strategy  
        """
    )

with col4:
    st.markdown("### 4️⃣ Iterate with new ideas")
    st.markdown(
        """
        Experiment with:

        - New hyperparameters inside each training script  
        - Different risk-aversion λ and transaction costs  
        - Alternate universes of tickers  
        - Different date ranges (longer or shorter memory)  
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
            "If you renamed your script, update the import paths above."
        )
        st.markdown(
            f"- **Sample window**: `{START}` → `{END}` (default)\n"
            "- **Tickers**: unknown\n"
        )

with conf_right:
    st.markdown("### Contents of a results `.pkl` file")

    st.markdown(
        """
        Each model (GBM / RF / PCA+LGBM / CatBoost) saves:

        - **`monthly`** — ML strategy & benchmark monthly returns  
        - **`weights`** — portfolio weights at each monthly rebalance  
        - **`monthly_train`** (2010–2023)  
        - **`monthly_test`** (2024–2025)  
        - **`weights_2025_bimonth`** — bi-monthly 2025 weights  
        - **`equity_2025`** — $1000 ML portfolio vs SPY (2025)  
        - **`metrics`** — Sharpe, vol, drawdown, etc.  
        """
    )

# ------------------------
# 3. Usage notes
# ------------------------
st.markdown("## 📝 Usage Notes")

st.markdown(
    textwrap.dedent(
        """
        - **Finance settings** (λ, transaction costs, RF rate) live on the
          **Run Model page**, not inside model scripts.

        - Results pages only **load `.pkl` files** — no retraining happens.

        - Every run creates a timestamped file:
            ```
            gradientboost_results_YYYYMMDD_HHMMSS.pkl  
            randomforest_results_YYYYMMDD_HHMMSS.pkl  
            pca_lightgbm_results_YYYYMMDD_HHMMSS.pkl  
            catboost_results_YYYYMMDD_HHMMSS.pkl
            ```
        - Keep multiple runs to compare hyperparameters or finance assumptions.
        """
    )
)

# ------------------------
# 4. Navigation hints
# ------------------------
st.markdown("## 🧭 Where to go next")

nav_left, nav_right = st.columns(2)

with nav_left:
    st.markdown("### ➡️ Run new backtests")
    st.markdown(
        """
        - Go to **“🧪 Run & Save ML Portfolio Backtests”**  
        - Choose GBM / RF / PCA+LGBM / CatBoost  
        - Run + save  
        """
    )

with nav_right:
    st.markdown("### 📊 Explore results")
    st.markdown(
        """
        - Visit each model’s results page  
        - Or compare all four in  
          **“⚖️ Model Comparison – GBM vs RF vs PCA+LGBM vs CatBoost”**  
        """
    )
