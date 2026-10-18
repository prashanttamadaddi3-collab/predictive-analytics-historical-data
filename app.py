from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "historical_sales.csv"
st.set_page_config(page_title="Predictive Analytics Studio", page_icon="📈", layout="wide")
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#f7f9ff,#eef4ff)}
.block-container{padding-top:1.5rem}
.hero{padding:1.4rem 1.6rem;border-radius:20px;background:linear-gradient(110deg,#173b78,#3568c8);color:white;margin-bottom:1rem}
.hero h1{color:white}.hero p{color:#e8efff}
[data-testid="stMetric"]{background:white;border:1px solid #dce6fa;border-radius:14px;padding:12px}
</style>
<div class="hero"><h1>📈 Predictive Analytics Studio</h1><p>Explore historical patterns, evaluate forecasting models, and estimate future trends.</p></div>
""", unsafe_allow_html=True)

def load_data(upload):
    df = pd.read_csv(upload if upload is not None else DEFAULT_DATA)
    date_col = next((c for c in df.columns if c.strip().lower() in ("date","month","timestamp","time")), None)
    target_col = next((c for c in df.columns if c.strip().lower() in ("sales","value","revenue","target","demand")), None)
    if not date_col or not target_col:
        raise ValueError("CSV needs a date column (Date/Month/Timestamp) and a numeric target (Sales/Revenue/Value).")
    df = df.rename(columns={date_col:"Date",target_col:"Target"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Target"] = pd.to_numeric(df["Target"], errors="coerce")
    df = df.dropna(subset=["Date","Target"]).sort_values("Date").groupby("Date",as_index=False)["Target"].mean()
    if len(df) < 12:
        raise ValueError("Provide at least 12 valid time points.")
    return df

def make_features(df):
    x = df.copy().sort_values("Date").reset_index(drop=True)
    x["t"] = np.arange(len(x))
    x["month_sin"] = np.sin(2*np.pi*x["Date"].dt.month/12)
    x["month_cos"] = np.cos(2*np.pi*x["Date"].dt.month/12)
    x["lag_1"] = x["Target"].shift(1)
    x["rolling_3"] = x["Target"].shift(1).rolling(3).mean()
    return x.dropna().reset_index(drop=True)

upload = st.sidebar.file_uploader("Upload historical CSV (optional)", type=["csv"])
horizon = st.sidebar.slider("Forecast horizon (months)", 1, 12, 6)
model_name = st.sidebar.selectbox("Model", ["Random Forest", "Linear Regression"])
st.sidebar.info("CSV should have monthly dates and a numeric Sales/Revenue/Value column.")
try:
    raw = load_data(upload)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

data = make_features(raw)
cols = ["t","month_sin","month_cos","lag_1","rolling_3"]
split = max(5, min(int(len(data)*.8), len(data)-2))
train, test = data.iloc[:split], data.iloc[split:]
def build_model():
    return RandomForestRegressor(n_estimators=250, random_state=42) if model_name=="Random Forest" else LinearRegression()
model = build_model()
model.fit(train[cols], train["Target"])
pred = model.predict(test[cols])
mae = mean_absolute_error(test["Target"], pred)
rmse = np.sqrt(mean_squared_error(test["Target"], pred))
r2 = r2_score(test["Target"], pred) if len(test)>1 else float("nan")

latest = float(raw.iloc[-1]["Target"])
previous = float(raw.iloc[-2]["Target"])
a,b,c,d = st.columns(4)
a.metric("Historical records", f"{len(raw):,}")
b.metric("Latest value", f"{latest:,.2f}", f"{latest-previous:+,.2f} vs prior")
c.metric("Test MAE", f"{mae:,.2f}")
d.metric("Test RMSE", f"{rmse:,.2f}")

t1,t2,t3,t4 = st.tabs(["Overview","Model Evaluation","Future Forecast","Data Preview"])
with t1:
    st.subheader("Historical trend")
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(raw["Date"],raw["Target"],marker="o",linewidth=2,label="Historical")
    ax.set_xlabel("Date");ax.set_ylabel("Target");ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    st.pyplot(fig)
    st.caption("Included data is synthetic demonstration data. Upload real data for meaningful analysis.")
with t2:
    st.subheader("Actual vs predicted on chronological test data")
    ev=test[["Date","Target"]].copy();ev["Predicted"]=pred
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(ev["Date"],ev["Target"],marker="o",label="Actual")
    ax.plot(ev["Date"],ev["Predicted"],marker="s",label="Predicted")
    ax.set_xlabel("Date");ax.set_ylabel("Target");ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    st.pyplot(fig)
    st.write(f"**R² score:** {r2:.3f}" if np.isfinite(r2) else "**R² score:** unavailable")
    st.dataframe(ev.round(2),use_container_width=True)
with t3:
    st.subheader(f"Next {horizon} periods")
    full=build_model();full.fit(data[cols],data["Target"])
    history=raw.copy().reset_index(drop=True);future=[]
    for _ in range(horizon):
        dt=history["Date"].iloc[-1]+pd.DateOffset(months=1)
        row={"t":len(history),"month_sin":np.sin(2*np.pi*dt.month/12),"month_cos":np.cos(2*np.pi*dt.month/12),
             "lag_1":float(history["Target"].iloc[-1]),"rolling_3":float(history["Target"].tail(3).mean())}
        y=float(full.predict(pd.DataFrame([row])[cols])[0])
        future.append({"Date":dt,"Forecast":y})
        history=pd.concat([history,pd.DataFrame([{"Date":dt,"Target":y}])],ignore_index=True)
    fdf=pd.DataFrame(future)
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(raw["Date"].tail(18),raw["Target"].tail(18),marker="o",label="Recent history")
    ax.plot(fdf["Date"],fdf["Forecast"],marker="s",linestyle="--",label="Forecast")
    ax.set_xlabel("Date");ax.set_ylabel("Target");ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    st.pyplot(fig)
    st.dataframe(fdf.assign(Forecast=fdf["Forecast"].round(2)),use_container_width=True)
    st.download_button("Download forecast CSV",fdf.to_csv(index=False).encode(),"forecast.csv","text/csv")
with t4:
    st.subheader("Cleaned dataset")
    st.dataframe(raw,use_container_width=True)
    st.download_button("Download cleaned CSV",raw.to_csv(index=False).encode(),"cleaned_historical_data.csv","text/csv")
