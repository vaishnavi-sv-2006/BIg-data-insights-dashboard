"""
Day 3: Sales Forecasting + Customer Segmentation
Big Data Insights Dashboard - E-commerce Star Schema

Part A: Monthly sales forecasting using Linear Regression (trend + seasonality)
Part B: Customer segmentation using RFM analysis + KMeans clustering
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error

DATA_DIR = "/home/claude/project3-big-data-insights/data"
np.random.seed(42)

df = pd.read_csv(f"{DATA_DIR}/sales_analytical_view.csv")
df["date"] = pd.to_datetime(df["date"])

print("=" * 60)
print("PART A: SALES FORECASTING")
print("=" * 60)

# ------------------------------------------------------------------
# Aggregate to monthly sales
# ------------------------------------------------------------------
monthly = df.groupby(pd.Grouper(key="date", freq="MS"))["total_amount"].sum().reset_index()
monthly = monthly.rename(columns={"total_amount": "revenue"})
monthly["month_index"] = np.arange(len(monthly))          # linear trend
monthly["month_num"] = monthly["date"].dt.month            # seasonality
monthly["quarter"] = monthly["date"].dt.quarter

# one-hot encode month for seasonality
month_dummies = pd.get_dummies(monthly["month_num"], prefix="m", drop_first=True)
X = pd.concat([monthly[["month_index"]], month_dummies], axis=1)
y = monthly["revenue"]

model = LinearRegression()
model.fit(X, y)
monthly["predicted"] = model.predict(X)

r2 = r2_score(y, monthly["predicted"])
mae = mean_absolute_error(y, monthly["predicted"])
print(f"Model fit on {len(monthly)} months")
print(f"R-squared: {r2:.3f}")
print(f"MAE: {mae:,.2f}")

# ------------------------------------------------------------------
# Forecast next 6 months
# ------------------------------------------------------------------
last_date = monthly["date"].max()
future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=6, freq="MS")
future = pd.DataFrame({"date": future_dates})
future["month_index"] = np.arange(len(monthly), len(monthly) + 6)
future["month_num"] = future["date"].dt.month
future_dummies = pd.get_dummies(future["month_num"], prefix="m", drop_first=True)
# align columns with training set (some months may be missing in future set)
future_dummies = future_dummies.reindex(columns=month_dummies.columns, fill_value=0)
X_future = pd.concat([future[["month_index"]], future_dummies], axis=1)
future["predicted_revenue"] = model.predict(X_future)
future["revenue"] = np.nan
future["type"] = "forecast"
monthly["type"] = "actual"
monthly["predicted_revenue"] = monthly["predicted"]

forecast_output = pd.concat([
    monthly[["date", "revenue", "predicted_revenue", "type"]],
    future[["date", "revenue", "predicted_revenue", "type"]]
], ignore_index=True)

print("\n--- Next 6-Month Forecast ---")
future_display = future[["date", "predicted_revenue"]].copy()
future_display["predicted_revenue"] = future_display["predicted_revenue"].round(2)
print(future_display.to_string(index=False))

forecast_output.to_csv(f"{DATA_DIR}/sales_forecast.csv", index=False)
print(f"\nSaved: sales_forecast.csv, shape={forecast_output.shape}")

# ==================================================================
print("\n" + "=" * 60)
print("PART B: CUSTOMER SEGMENTATION (RFM + KMeans)")
print("=" * 60)

# ------------------------------------------------------------------
# Compute RFM metrics per customer
# ------------------------------------------------------------------
snapshot_date = df["date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("customer_id").agg(
    recency=("date", lambda x: (snapshot_date - x.max()).days),
    frequency=("sale_id", "count"),
    monetary=("total_amount", "sum")
).reset_index()

print(f"\nRFM computed for {len(rfm)} customers")
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

# ------------------------------------------------------------------
# Scale + KMeans clustering (4 segments)
# ------------------------------------------------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

# label clusters based on average RFM behavior (lower recency + higher freq/monetary = better)
cluster_profile = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
cluster_profile["score"] = (
    -cluster_profile["recency"].rank() + cluster_profile["frequency"].rank() + cluster_profile["monetary"].rank()
)
ranked = cluster_profile.sort_values("score", ascending=False).index.tolist()

labels = ["Champions", "Loyal Customers", "At Risk", "New / Low Engagement"]
label_map = {cluster_id: labels[i] for i, cluster_id in enumerate(ranked)}
rfm["segment"] = rfm["cluster"].map(label_map)

print("\n--- Segment Profile (avg RFM per segment) ---")
segment_summary = rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(2)
segment_summary["customer_count"] = rfm["segment"].value_counts()
print(segment_summary)

rfm.to_csv(f"{DATA_DIR}/customer_segments.csv", index=False)
print(f"\nSaved: customer_segments.csv, shape={rfm.shape}")

print("\n" + "=" * 60)
print("DAY 3 COMPLETE")
print("=" * 60)
