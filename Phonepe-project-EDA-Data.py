# 📦 Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
from sklearn.preprocessing import LabelEncoder

# 📂 Manually Load Dataset
data = {
    "state": ["Maharashtra", "Karnataka", "Delhi", "Uttar Pradesh", "Gujarat", "Rajasthan", "Punjab", "Bihar"],
    "year": [2020, 2021, 2020, 2022, 2021, 2022, 2020, 2021],
    "quarter": [1, 2, 3, 4, 2, 1, 4, 3],
    "transaction_type": [
        "Recharge & bill payments", "Peer-to-peer payments", "Merchant payments", "Financial Services",
        "Others", "Merchant payments", "Peer-to-peer payments", "Recharge & bill payments"
    ],
    "transaction_count": [150000, 250000, 175000, 300000, 200000, 180000, 225000, 160000],
    "transaction_amount": [50000000, 75000000, 62000000, 90000000, 68000000, 59000000, 71000000, 52000000]
}

df = pd.DataFrame(data)

# 👁️ Dataset First Look
print(df.head())

# 🔢 Rows & Columns
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# ℹ️ Dataset Info
df.info()

# ❌ Check Duplicates & Missing Values
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:\n", df.isnull().sum())

# 🔧 Prepare Dataset for Analysis
le = LabelEncoder()
df["transaction_type_encoded"] = le.fit_transform(df["transaction_type"])

# 📊 Dataset Describe
print(df.describe())

# 📈 Chart 1 - Year-wise Transaction Amount
yearly = df.groupby("year")["transaction_amount"].sum().reset_index()
sns.barplot(data=yearly, x="year", y="transaction_amount", palette="Blues")
plt.title("Year-wise Total Transaction Amount")
plt.show()

# 📊 Chart 2 - Transaction Type vs Amount
type_data = df.groupby("transaction_type")["transaction_amount"].sum().reset_index().sort_values(by="transaction_amount", ascending=False)
sns.barplot(data=type_data, x="transaction_type", y="transaction_amount", palette="Set2")
plt.xticks(rotation=45)
plt.title("Transaction Amount by Transaction Type")
plt.show()

# 📍 Chart 3 - Count vs Amount
sns.scatterplot(data=df, x="transaction_count", y="transaction_amount", hue="transaction_type", s=100)
plt.title("Transaction Count vs Amount")
plt.grid(True)
plt.show()

# 📉 Chart 4 - Quarter-wise Transaction Amount
q_amt = df.groupby("quarter")["transaction_amount"].sum().reset_index()
sns.lineplot(data=q_amt, x="quarter", y="transaction_amount", marker="o", color="orange")
plt.title("Quarter-wise Transaction Amount")
plt.show()

# 📊 Chart 5 - Transaction Type vs Count
type_count = df.groupby("transaction_type")["transaction_count"].sum().reset_index()
sns.barplot(data=type_count, x="transaction_type", y="transaction_count", palette="coolwarm")
plt.title("Transaction Count by Type")
plt.xticks(rotation=45)
plt.show()

# 📉 Chart 6 - Quarter-wise Count
q_count = df.groupby("quarter")["transaction_count"].sum().reset_index()
sns.lineplot(data=q_count, x="quarter", y="transaction_count", marker="o", color="green")
plt.title("Quarter-wise Transaction Count")
plt.show()

# 📊 Chart 7 - Year-wise Count
yearly_count = df.groupby("year")["transaction_count"].sum().reset_index()
sns.barplot(data=yearly_count, x="year", y="transaction_count", palette="BuGn")
plt.title("Year-wise Transaction Count")
plt.show()

# 📍 Chart 8 - State-wise Transaction Amount
state_amt = df.groupby("state")["transaction_amount"].sum().reset_index().sort_values(by="transaction_amount", ascending=False)
sns.barplot(data=state_amt, x="transaction_amount", y="state", palette="flare")
plt.title("State-wise Transaction Amount")
plt.show()

# 📍 Chart 9 - State-wise Transaction Count
state_count = df.groupby("state")["transaction_count"].sum().reset_index().sort_values(by="transaction_count", ascending=False)
sns.barplot(data=state_count, x="transaction_count", y="state", palette="crest")
plt.title("State-wise Transaction Count")
plt.show()

# 🔥 Correlation Heatmap
corr = df.select_dtypes(include=["int64", "float64"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 📊 Pair Plot
sns.pairplot(df.select_dtypes(include=["int64", "float64"]), diag_kind="kde", corner=True)
plt.suptitle("Pair Plot of Numerical Features", y=1.02)
plt.show()
