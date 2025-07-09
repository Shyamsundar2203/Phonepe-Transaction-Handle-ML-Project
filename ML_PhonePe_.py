#  1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 2. Define Manual Dataset


data = {
    "state": ["Maharashtra", "Karnataka", "Delhi", "Uttar Pradesh", "Gujarat"],
    "year": [2020, 2021, 2020, 2022, 2021],
    "quarter": [1, 2, 3, 4, 2],
    "transaction_type": ["Recharge & bill payments", "Peer-to-peer payments", "Merchant payments", "Financial Services", "Others"],
    "transaction_count": [150000, 250000, 175000, 300000, 200000],
    "transaction_amount": [50000000, 75000000, 62000000, 90000000, 68000000]
}

df = pd.DataFrame(data)
print(df)
#  3. Preprocessing


le = LabelEncoder()
df['transaction_type_encoded'] = le.fit_transform(df['transaction_type'])

df_model = df.drop(['transaction_type', 'state'], axis=1)

X = df_model.drop("transaction_amount", axis=1)
y = df_model["transaction_amount"]

#  4. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Models

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 6. Evaluate

print("\n📈 Linear Regression:")
print("MAE:", mean_absolute_error(y_test, lr_preds))
print("R2 Score:", r2_score(y_test, lr_preds))

print("\n🌲 Random Forest:")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("R2 Score:", r2_score(y_test, rf_preds))

# 7. Plot Actual vs Predicted

plt.figure(figsize=(8, 4))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(rf_preds, label="Predicted", marker='x')
plt.title("Actual vs Predicted - Random Forest")
plt.xlabel("Samples")
plt.ylabel("Transaction Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
