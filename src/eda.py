import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(os.path.join("data", "cleaned_retail.csv"))

os.makedirs("notebooks/eda_plots", exist_ok=True)

# top 20 products
plt.figure(figsize=(12, 6))
top_products = df["Description"].value_counts().head(20)
sns.barplot(x=top_products.values, y=top_products.index, palette="Blues_r")
plt.title("top 20 most purchased products")
plt.xlabel("frequency")
plt.tight_layout()
plt.savefig("notebooks/eda_plots/top_products.png", dpi=150)
plt.show()
print("saved top_products.png")

# sales by country
plt.figure(figsize=(12, 6))
top_countries = df["Country"].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, palette="Oranges_r")
plt.title("top 10 countries by transaction count")
plt.xlabel("transactions")
plt.tight_layout()
plt.savefig("notebooks/eda_plots/top_countries.png", dpi=150)
plt.show()
print("saved top_countries.png")

# revenue distribution
df["Revenue"] = df["Quantity"] * df["Price"]
plt.figure(figsize=(10, 4))
df_uk = df[df["Country"] == "United Kingdom"]
df_uk["Revenue"].clip(upper=df_uk["Revenue"].quantile(0.99)).hist(bins=60, color="#2196F3", edgecolor="white")
plt.title("revenue distribution (uk, clipped at 99th percentile)")
plt.xlabel("revenue per line item")
plt.tight_layout()
plt.savefig("notebooks/eda_plots/revenue_dist.png", dpi=150)
plt.show()
print("saved revenue_dist.png")

# monthly sales trend
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Month"] = df["InvoiceDate"].dt.to_period("M")
monthly = df.groupby("Month")["Revenue"].sum().reset_index()
monthly["Month"] = monthly["Month"].astype(str)

plt.figure(figsize=(12, 4))
plt.plot(monthly["Month"], monthly["Revenue"], marker="o", color="#E91E63")
plt.xticks(rotation=45, ha="right")
plt.title("monthly revenue trend")
plt.tight_layout()
plt.savefig("notebooks/eda_plots/monthly_trend.png", dpi=150)
plt.show()
print("saved monthly_trend.png")

print("\neda complete. all plots saved to notebooks/eda_plots/")