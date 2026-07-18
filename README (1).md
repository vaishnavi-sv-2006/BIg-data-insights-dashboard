# Big Data Insights Dashboard

An advanced-level analytics project simulating a multi-region e-commerce data warehouse using a **star schema**, Python-based data engineering, machine learning (sales forecasting + customer segmentation), and an interactive Power BI dashboard.

This is Project 3 of 4 in a structured data analytics portfolio built for CodeTech certification.

---

## Project structure

```
project3-big-data-insights/
├── data/
│   ├── dim_region.csv
│   ├── dim_customers.csv
│   ├── dim_products.csv
│   ├── dim_date.csv
│   ├── fact_sales.csv
│   ├── dim_region_clean.csv
│   ├── dim_customers_clean.csv
│   ├── dim_products_clean.csv
│   ├── dim_date_clean.csv
│   ├── fact_sales_clean.csv
│   ├── sales_analytical_view.csv
│   ├── sales_forecast.csv
│   └── customer_segments.csv
├── script/
│   ├── day1_generate_data.py
│   ├── day2_clean_eda.py
│   └── day3_forecast_segmentation.py
├── BigDataInsightsDashboard.pbix
├── LICENSE
└── README.md
```

---

## Workflow

### Day 1 — Data generation
Generated a **star schema** dataset simulating a multi-region e-commerce platform:
- `fact_sales` — 150,000+ transaction records
- `dim_customers` — 20,000+ customers
- `dim_products` — 500 products across 5 categories
- `dim_region` — 6 regions across India
- `dim_date` — full calendar table, 2022–2024

Intentional data quality issues were injected for realism: null values, duplicate rows, inconsistent text casing, and price/revenue outliers.

### Day 2 — Cleaning & exploratory data analysis
- Removed duplicate rows and standardized casing across all text fields
- Handled nulls via median imputation and recomputation from related fields
- Capped statistical outliers using the IQR method
- Consolidated inconsistent category labels (e.g. payment modes)
- Built a joined analytical view across all 5 tables for EDA

### Day 3 — Machine learning
**Sales forecasting:** Linear Regression model using trend + monthly seasonality features, forecasting the next 6 months of revenue (R² = 0.664).

**Customer segmentation:** RFM (Recency, Frequency, Monetary) analysis combined with K-Means clustering to segment ~20,000 customers into 4 behavioral groups — Champions, Loyal Customers, At Risk, and New/Low Engagement.

### Day 4 — Power BI dashboard
Built a 4-page interactive dashboard on a proper star-schema data model (not a flattened table), with custom DAX measures for revenue, growth, and segment-level KPIs:
- Executive Overview
- Category & Product Analysis
- Customer Segmentation
- Sales Forecast

---

## Key metrics

| Metric | Value |
|---|---|
| Total transactions | 150,000 |
| Total customers | 20,000 |
| Total revenue (2022–2024) | ~₹15.7B |
| Forecast horizon | 6 months (Jan–Jun 2025) |
| Forecast model R² | 0.664 |
| Customer segments identified | 4 (Champions, Loyal, At Risk, New) |
| Champions segment avg. spend | ₹13,41,648 |

---

## Tech stack

- **Python** — pandas, numpy, scikit-learn (data generation, cleaning, forecasting, clustering)
- **Power BI Desktop** — star-schema data modeling, DAX, interactive dashboard
- **GitHub** — version control and portfolio hosting

---

## Portfolio tracker

| # | Project | Level | Status |
|---|---|---|---|
| 1 | Sales Trend Dashboard | Basic | ✅ Complete |
| 2 | Customer Churn Prediction | Intermediate | ✅ Complete |
| 3 | Big Data Insights Dashboard | Advanced | ✅ Complete |
| 4 | AI-Powered Business Intelligence Dashboard | Pro | ⏳ Upcoming |
