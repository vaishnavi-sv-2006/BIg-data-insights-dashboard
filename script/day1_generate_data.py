"""
Day 1: Generate Big Data Insights Dashboard dataset
Star schema: fact_sales (150k+ rows) + 4 dimension tables
Domain: Multi-region E-commerce
Includes intentional data quality issues for realistic cleaning practice.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

OUT_DIR = "/home/claude/project3-big-data-insights/data"

# ------------------------------------------------------------------
# 1. DIM_REGION
# ------------------------------------------------------------------
regions = [
    ("R01", "North India", "India", "North"),
    ("R02", "South India", "India", "South"),
    ("R03", "East India", "India", "East"),
    ("R04", "West India", "India", "West"),
    ("R05", "Central India", "India", "Central"),
    ("R06", "North East", "India", "North East"),
]
dim_region = pd.DataFrame(regions, columns=["region_id", "region_name", "country", "zone"])
# inconsistent casing issue
dim_region.loc[2, "region_name"] = "east india"
dim_region.loc[4, "zone"] = "CENTRAL"

# ------------------------------------------------------------------
# 2. DIM_CUSTOMERS  (20,000 customers)
# ------------------------------------------------------------------
n_customers = 20000
first_names = ["Arun", "Divya", "Karthik", "Priya", "Suresh", "Anitha", "Ramesh", "Kavya",
               "Vijay", "Meena", "Ganesh", "Lakshmi", "Manoj", "Deepa", "Sathish", "Nithya"]
last_names = ["Kumar", "Raj", "Prakash", "Devi", "Murthy", "Iyer", "Reddy", "Nair",
              "Pillai", "Sharma", "Gupta", "Rao"]
cities = ["Chennai", "chennai", "CHENNAI", "Mumbai", "Delhi", "delhi", "Bangalore",
          "Kolkata", "Hyderabad", "Pune", "Coimbatore", "Ahmedabad", "Jaipur", "Lucknow"]
states = ["Tamil Nadu", "Maharashtra", "Delhi", "Karnataka", "West Bengal",
          "Telangana", "Gujarat", "Rajasthan", "Uttar Pradesh"]
segments = ["Regular", "Premium", "New", "VIP"]

customer_ids = [f"C{100000+i}" for i in range(n_customers)]
cust_data = {
    "customer_id": customer_ids,
    "customer_name": [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_customers)],
    "email": [f"user{i}@mail.com" for i in range(n_customers)],
    "city": [random.choice(cities) for _ in range(n_customers)],
    "state": [random.choice(states) for _ in range(n_customers)],
    "signup_date": [
        (datetime(2021, 1, 1) + timedelta(days=random.randint(0, 1460))).strftime("%Y-%m-%d")
        for _ in range(n_customers)
    ],
    "customer_segment": [random.choice(segments) for _ in range(n_customers)],
}
dim_customers = pd.DataFrame(cust_data)

# inject nulls
null_idx = np.random.choice(n_customers, size=int(n_customers * 0.03), replace=False)
dim_customers.loc[null_idx, "email"] = np.nan
null_idx2 = np.random.choice(n_customers, size=int(n_customers * 0.02), replace=False)
dim_customers.loc[null_idx2, "city"] = np.nan

# inject duplicates
dupes = dim_customers.sample(150, random_state=1)
dim_customers = pd.concat([dim_customers, dupes], ignore_index=True)

# ------------------------------------------------------------------
# 3. DIM_PRODUCTS  (500 products)
# ------------------------------------------------------------------
categories = {
    "Electronics": ["Mobiles", "Laptops", "Accessories", "Cameras"],
    "Fashion": ["Men Clothing", "Women Clothing", "Footwear", "Watches"],
    "Home": ["Furniture", "Kitchen", "Decor", "Appliances"],
    "Grocery": ["Snacks", "Beverages", "Staples", "Personal Care"],
    "Sports": ["Fitness", "Outdoor", "Team Sports", "Apparel"],
}
n_products = 500
product_rows = []
for i in range(n_products):
    cat = random.choice(list(categories.keys()))
    sub = random.choice(categories[cat])
    price = round(np.random.uniform(99, 45000), 2)
    product_rows.append((f"P{2000+i}", f"{sub} Item {i}", cat, sub, price))
dim_products = pd.DataFrame(product_rows, columns=["product_id", "product_name", "category", "sub_category", "unit_price"])

# casing inconsistency
dim_products.loc[10:15, "category"] = dim_products.loc[10:15, "category"].str.upper()
dim_products.loc[20:25, "category"] = dim_products.loc[20:25, "category"].str.lower()

# a few negative/zero price outliers (data entry errors)
outlier_idx = np.random.choice(n_products, size=5, replace=False)
dim_products.loc[outlier_idx, "unit_price"] = [-100, 0, -50, 999999, 0]

# ------------------------------------------------------------------
# 4. DIM_DATE  (3 years: 2022-01-01 to 2024-12-31)
# ------------------------------------------------------------------
date_range = pd.date_range("2022-01-01", "2024-12-31", freq="D")
dim_date = pd.DataFrame({
    "date_id": date_range.strftime("%Y%m%d").astype(int),
    "date": date_range.strftime("%Y-%m-%d"),
    "day": date_range.day,
    "month": date_range.month,
    "month_name": date_range.strftime("%B"),
    "quarter": date_range.quarter,
    "year": date_range.year,
    "is_weekend": date_range.dayofweek.isin([5, 6]),
})

# ------------------------------------------------------------------
# 5. FACT_SALES  (~150,000 rows) - the "big data" fact table
# ------------------------------------------------------------------
n_facts = 150000
payment_modes = ["Credit Card", "UPI", "Debit Card", "Net Banking", "Cash on Delivery",
                  "credit card", "COD"]  # inconsistent casing/labels intentional

fact_customer_ids = np.random.choice(dim_customers["customer_id"].dropna().unique(), n_facts)
fact_product_ids = np.random.choice(dim_products["product_id"], n_facts)
fact_region_ids = np.random.choice(dim_region["region_id"], n_facts)
fact_date_ids = np.random.choice(dim_date["date_id"], n_facts)

quantities = np.random.randint(1, 10, n_facts)
# merge unit prices to compute realistic total_amount
price_lookup = dim_products.set_index("product_id")["unit_price"].to_dict()
unit_prices = np.array([price_lookup.get(pid, 0) for pid in fact_product_ids])
discounts = np.round(np.random.choice([0, 0.05, 0.1, 0.15, 0.2, 0.25], n_facts,
                                       p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.1]), 2)
total_amount = np.round(quantities * unit_prices * (1 - discounts), 2)

fact_sales = pd.DataFrame({
    "sale_id": [f"S{1000000+i}" for i in range(n_facts)],
    "date_id": fact_date_ids,
    "customer_id": fact_customer_ids,
    "product_id": fact_product_ids,
    "region_id": fact_region_ids,
    "quantity": quantities,
    "discount": discounts,
    "total_amount": total_amount,
    "payment_mode": np.random.choice(payment_modes, n_facts),
})

# inject nulls in quantity / total_amount
null_idx3 = np.random.choice(n_facts, size=int(n_facts * 0.02), replace=False)
fact_sales.loc[null_idx3, "total_amount"] = np.nan
null_idx4 = np.random.choice(n_facts, size=int(n_facts * 0.01), replace=False)
fact_sales.loc[null_idx4, "quantity"] = np.nan

# inject outliers (extreme total_amount)
outlier_idx2 = np.random.choice(n_facts, size=30, replace=False)
fact_sales.loc[outlier_idx2, "total_amount"] = fact_sales.loc[outlier_idx2, "total_amount"] * 50

# inject duplicate rows
dupe_rows = fact_sales.sample(500, random_state=2)
fact_sales = pd.concat([fact_sales, dupe_rows], ignore_index=True)

# shuffle
fact_sales = fact_sales.sample(frac=1, random_state=3).reset_index(drop=True)

# ------------------------------------------------------------------
# SAVE ALL FILES
# ------------------------------------------------------------------
dim_region.to_csv(f"{OUT_DIR}/dim_region.csv", index=False)
dim_customers.to_csv(f"{OUT_DIR}/dim_customers.csv", index=False)
dim_products.to_csv(f"{OUT_DIR}/dim_products.csv", index=False)
dim_date.to_csv(f"{OUT_DIR}/dim_date.csv", index=False)
fact_sales.to_csv(f"{OUT_DIR}/fact_sales.csv", index=False)

print("Files generated successfully:")
print(f"  dim_region.csv     : {dim_region.shape}")
print(f"  dim_customers.csv  : {dim_customers.shape}")
print(f"  dim_products.csv   : {dim_products.shape}")
print(f"  dim_date.csv       : {dim_date.shape}")
print(f"  fact_sales.csv     : {fact_sales.shape}")
print(f"\nTotal fact_sales rows: {len(fact_sales):,}")
