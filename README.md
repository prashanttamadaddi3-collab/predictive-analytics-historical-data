# Predictive Analytics Using Historical Data

Interactive forecasting dashboard built with Python and Streamlit.

## Features
- Historical trend visualization
- Upload a CSV with a date column and numeric target
- Data cleaning and time-based feature engineering
- Linear Regression and Random Forest forecasting models
- Time-ordered holdout evaluation using MAE, RMSE, and R²
- Forecast future periods and download forecasts as CSV

## Run locally
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The included `data/historical_sales.csv` is synthetic demo data, not real company data. CSV uploads should include a date column (`Date`, `Month`, or `Timestamp`) and numeric target (`Sales`, `Revenue`, `Value`, `Demand`, or `Target`). Monthly data is recommended.

Forecasts are estimates for learning and demonstration, not business guarantees.
