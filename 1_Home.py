import textwrap

import streamlit as st

# Optional: pull config from your training script if it's importable
try:
    import _gradientboost as gb  # or gradientboost_train if you used that name

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

st.title("🔮 Gradient Boosted Portfolio Lab")

st.markdown(
    """
This app trains a **gradient-boosted stock selection model** and compares its 
optimized portfolio against the **S&P 500**, as if you invested \$1000 in each.

Use this home page as your **control panel**, and the results page to **inspect
saved runs**.
"""
)

# ------------------------
# 1. How the workflow works
# ------------------------
st.markdown("## 🚦 Workflow Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ Train the model (offline)")
    st.markdown(
        """
        Run the training script once to:
        - Download historical prices  
        - Train the gradient boosting model (GPU-accelerated if available)  
        - Run the portfolio backtest  
        - Save everything into a single `.pkl` results file  
        """
    )
    st.code(
        "python _gradientboost.py",
        language="bash",
    )
    st.caption(
        "You can rename the script if you like—just keep the same main block "
        "that saves a `gradientboost_results_*.pkl` file in a `results/` folder."
    )

with col2:
    st.markdown("### 2️⃣ Load results in the dashboard")
    st.markdown(
        """
        Once a run has finished, you'll have a file like:
        - `results/gradientboost_results_YYYYMMDD_HHMMSS.pkl`
        
        Go to the **“📈 Gradient Boost Portfolio – Backtest Results”** page and:
        - Upload that `.pkl` file  
        - View equity curves for 2025 (\$1000 ML vs \$1000 S&P)  
        - Inspect portfolio weights every 2 months in 2025  
        - Browse training & testing metrics  
        """
    )

with col3:
    st.markdown("### 3️⃣ Iterate on design ideas")
    st.markdown(
        """
        You can experiment with:
        - Different ticker universes  
        - Training windows (e.g., 2010–2024)  
        - Risk-aversion and optimization settings  
        - Rebalance frequencies and transaction costs  
        
        Just update `_gradientboost.py`, rerun training, and upload the new results file.
        """
    )

# ------------------------
# 2. Current configuration
# ------------------------
st.markdown("## ⚙️ Current Configuration (from training script)")

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
            "- **Tickers**: unknown (set in your training script)\n"
        )

with conf_right:
    st.markdown("### Outputs per training run")

    st.markdown(
        """
        Each run of `_gradientboost.py` saves a results dictionary containing:
        
        - **`monthly`**: monthly strategy and benchmark returns  
        - **`weights`**: portfolio weights at each rebalance date  
        - **`monthly_train` / `monthly_test`**: split returns for train vs test  
        - **`weights_2025_bimonth`**: weights every 2 months in 2025  
        - **`equity_2025`**: \$1000 ML portfolio vs \$1000 S&P 500  
        - **`metrics`**: training/testing performance metrics  
        """
    )

# ------------------------
# 3. Quick usage notes
# ------------------------
st.markdown("## 📝 Usage Notes")

st.markdown(
    textwrap.dedent(
        """
        - **No auto-retraining in the dashboard**  
          The Streamlit pages only **load and visualize** saved results. This keeps the UI fast and ensures
          you don’t accidentally kick off long retraining runs just by refreshing the app.
        
        - **GPU acceleration**  
          If `xgboost` with GPU support is installed, `_gradientboost.py` will automatically use GPU-backed
          training. If not, it falls back to a standard CPU-based gradient boosting model.
        
        - **Multiple runs & versioning**  
          Each training run writes a new `.pkl` with a timestamp in the filename. You can keep multiple runs
          side-by-side (e.g., different universes or hyperparameters) and upload whichever one you want
          to inspect.
        """
    )
)

# ------------------------
# 4. Quick links / navigation hints
# ------------------------
st.markdown("## 🧭 Where to go next")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown("### ➡️ View saved runs")
    st.markdown(
        """
        Go to the **“📈 Gradient Boost Portfolio – Backtest Results”** page in the sidebar to:
        
        - Upload a `.pkl` results file  
        - Compare ML vs S&P 500 in 2025  
        - Inspect portfolio weights and composition over time  
        - Review training/testing performance metrics  
        """
    )

with nav_col2:
    st.markdown("### 🛠 Customize the model")
    st.markdown(
        """
        Edit `_gradientboost.py` to tweak:
        
        - The ticker list (`TICKERS`)  
        - Date range (`START`, `END`)  
        - Risk-aversion parameter (`LAMBDA_RISK`)  
        - Rebalance frequency and transaction cost assumptions  
        
        Then rerun:
        """
    )
    st.code("python _gradientboost.py", language="bash")
