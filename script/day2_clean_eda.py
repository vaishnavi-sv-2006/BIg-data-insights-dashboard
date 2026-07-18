"""
Day 2: Clean all 5 tables + Exploratory Data Analysis
Big Data Insights Dashboard - E-commerce Star Schema
"""

import pandas as pd
import numpy as np

DATA_DIR = "/home/claude/project3-big-data-insights/data"

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

# ==================================================================
# LOAD RAW DATA
# ==================================================================
dim_region = pd.read_csv(f"{DATA_DIR}/dim_region.csv")
dim_customers = pd.read_csv(f"{DATA_DIR}/dim_customers.csv")
dim_products = pd.read_csv(f"{DATA_DIR}/dim_products.csv")
dim_date = pd.read_csv(f"{DATA_DIR}/dim_date.csv")
fact_sales = pd.read_csv(f"{DATA_DIR}/fact_sales.csv")

print("=" * 60)
print("RAW SHAPES")
print("=" * 60)
for name, df in [("dim_region", dim_region), ("dim_customers", dim_customers),
                  ("dim_products", dim_products), ("dim_date", dim_date),
                  ("fact_sales", fact_sales)]:
    print(f"{name:15s}: {df.shape}")

# ==================================================================
# CLEAN: dim_region
# ==================================================================
dim_region["region_name"] = dim_region["region_name"].str.title()
dim_region["zone"] = dim_region["zone"].str.title()

# ==================================================================
# CLEAN: dim_customers
# ==================================================================
dim_customers = dim_customers.drop_duplicates()
dim_customers["city"] = dim_customers["city"].str.strip().str.title()
dim_customers["city"] = dim_customers["city"].fillna("Unknown")
dim_customers["email"] = dim_customers["email"].fillna("not_provided@mail.com")
dim_customers["customer_name"] = dim_customers["customer_name"].str.strip().str.title()

# ==================================================================
# CLEAN: dim_products
# ==================================================================
dim_products["category"] = dim_products["category"].str.strip().str.title()
dim_products["sub_category"] = dim_products["sub_category"].str.strip().str.title()

# fix invalid prices: replace <=0 or absurdly high with category median
dim_products.loc[dim_products["unit_price"] <= 0, "unit_price"] = np.nan
cat_median = dim_products.groupby("category")["unit_price"].transform("median")
dim_products["unit_price"] = dim_products["unit_price"].fillna(cat_median)
# cap extreme high outlier (999999 was a data entry error)
price_cap = dim_products["unit_price"].quantile(0.99)
dim_products.loc[dim_products["unit_price"] > price_cap, "unit_price"] = price_cap

# ==================================================================
# CLEAN: fact_sales
# ==================================================================
fact_sales = fact_sales.drop_duplicates()

# standardize payment_mode labels
payment_map = {
    "credit card": "Credit Card",
    "Credit Card": "Credit Card",
    "COD": "Cash on Delivery",
    "Cash on Delivery": "Cash on Delivery",
    "UPI": "UPI",
    "Debit Card": "Debit Card",
    "Net Banking": "Net Banking",
}
fact_sales["payment_mode"] = fact_sales["payment_mode"].map(payment_map)

# handle null quantity -> fill with median quantity
fact_sales["quantity"] = fact_sales["quantity"].fillna(fact_sales["quantity"].median())

# handle null total_amount -> recompute from product price where possible
price_lookup = dim_products.set_index("product_id")["unit_price"].to_dict()
missing_amt = fact_sales["total_amount"].isna()
fact_sales.loc[missing_amt, "total_amount"] = fact_sales.loc[missing_amt].apply(
    lambda r: r["quantity"] * price_lookup.get(r["product_id"], 0) * (1 - r["discount"]), axis=1
)

# cap extreme outliers in total_amount (the x50 injected outliers) using IQR method
Q1 = fact_sales["total_amount"].quantile(0.25)
Q3 = fact_sales["total_amount"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 3 * IQR
outlier_count = (fact_sales["total_amount"] > upper_bound).sum()
fact_sales.loc[fact_sales["total_amount"] > upper_bound, "total_amount"] = upper_bound

print(f"\nOutliers capped in fact_sales.total_amount: {outlier_count}")

# ==================================================================
# SAVE CLEANED FILES
# ==================================================================
dim_region.to_csv(f"{DATA_DIR}/dim_region_clean.csv", index=False)
dim_customers.to_csv(f"{DATA_DIR}/dim_customers_clean.csv", index=False)
dim_products.to_csv(f"{DATA_DIR}/dim_products_clean.csv", index=False)
dim_date.to_csv(f"{DATA_DIR}/dim_date_clean.csv", index=False)
fact_sales.to_csv(f"{DATA_DIR}/fact_sales_clean.csv", index=False)

print("\n" + "=" * 60)
print("CLEANED SHAPES")
print("=" * 60)
for name, df in [("dim_region", dim_region), ("dim_customers", dim_customers),
                  ("dim_products", dim_products), ("dim_date", dim_date),
                  ("fact_sales", fact_sales)]:
    print(f"{name:15s}: {df.shape}")

# ==================================================================
# BUILD ANALYTICAL VIEW (join fact + dims) FOR EDA
# ==================================================================
df = fact_sales.merge(dim_customers[["customer_id", "city", "state", "customer_segment"]],
                       on="customer_id", how="left")
df = df.merge(dim_products[["product_id", "category", "sub_category", "unit_price"]],
              on="product_id", how="left")
df = df.merge(dim_region[["region_id", "region_name", "zone"]], on="region_id", how="left")
df = df.merge(dim_date[["date_id", "date", "month_name", "quarter", "year", "is_weekend"]],
              on="date_id", how="left")

print("\n" + "=" * 60)
print("EDA: KEY INSIGHTS")
print("=" * 60)

print("\n--- Total Revenue by Region ---")
print(df.groupby("region_name")["total_amount"].sum().sort_values(ascending=False).round(2))

print("\n--- Total Revenue by Category ---")
print(df.groupby("category")["total_amount"].sum().sort_values(ascending=False).round(2))

print("\n--- Revenue by Year ---")
print(df.groupby("year")["total_amount"].sum().round(2))

print("\n--- Top 5 Customer Segments by Revenue ---")
print(df.groupby("customer_segment")["total_amount"].sum().sort_values(ascending=False).round(2))

print("\n--- Payment Mode Distribution ---")
print(df["payment_mode"].value_counts())

print("\n--- Weekend vs Weekday Revenue ---")
print(df.groupby("is_weekend")["total_amount"].sum().round(2))

print("\n--- Top 10 Products by Revenue ---")
top_products = df.groupby("product_id")["total_amount"].sum().sort_values(ascending=False).head(10)
print(top_products.round(2))

# Save analytical view for Day 3 Python modeling + Power BI reference
df.to_csv(f"{DATA_DIR}/sales_analytical_view.csv", index=False)
print(f"\nAnalytical (joined) view saved: sales_analytical_view.csv, shape={df.shape}")
