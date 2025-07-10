# app.py - Streamlit Dashboard for PhonePe Transaction Insights

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="PhonePe Transaction Insights", layout="wide")
st.title("📊 PhonePe Transaction Insights Dashboard")
st.markdown("Analyze transaction trends across states, years, and transaction types.")

# Load Data
@st.cache_data
def load_data():
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
    return pd.DataFrame(data)

df = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filter Data")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()))
selected_quarter = st.sidebar.selectbox("Select Quarter", sorted(df["quarter"].unique()))
selected_type = st.sidebar.selectbox("Select Transaction Type", df["transaction_type"].unique())

filtered_df = df[
    (df["year"] == selected_year) &
    (df["quarter"] == selected_quarter) &
    (df["transaction_type"] == selected_type)
]

# KPI Display
st.subheader("🔢 Key Stats")
col1, col2 = st.columns(2)
col1.metric("Total Transaction Count", f'{filtered_df["transaction_count"].sum():,}')
col2.metric("Total Transaction Amount", f'₹ {filtered_df["transaction_amount"].sum():,}')

# Chart - Transaction Amount by State
st.subheader("📍 Transaction Amount by State")
state_chart = df.groupby("state")["transaction_amount"].sum().reset_index().sort_values(by="transaction_amount", ascending=False)
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(data=state_chart, y="state", x="transaction_amount", palette="magma", ax=ax1)
ax1.set_xlabel("Transaction Amount (₹)")
st.pyplot(fig1)

# Chart - Transaction Count by Type
st.subheader("🧾 Transaction Count by Type")
type_chart = df.groupby("transaction_type")["transaction_count"].sum().reset_index()
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.barplot(data=type_chart, x="transaction_type", y="transaction_count", palette="coolwarm", ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("📌 Built with Streamlit | Data based on PhonePe Pulse | Project by [Your Name]")
