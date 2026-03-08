import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
os.makedirs("outputs", exist_ok=True)
# ── Step 1: Load ──────────────────────────────────────────
df = pd.read_excel("data/Online Retail.xlsx")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Step 2: Clean ─────────────────────────────────────────
df.dropna(subset=['CustomerID', 'Description'], inplace=True)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['Revenue'] = df['Quantity'] * df['UnitPrice']
print(f"After cleaning: {df.shape[0]} rows")

# ── Step 3: Feature Engineering ───────────────────────────
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year']        = df['InvoiceDate'].dt.year
df['Month']       = df['InvoiceDate'].dt.month
df['Week']        = df['InvoiceDate'].dt.isocalendar().week.astype(int)
df['DayOfWeek']   = df['InvoiceDate'].dt.dayofweek

monthly_sales = df.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
monthly_sales['MonthIndex'] = range(len(monthly_sales))
print("\nMonthly Sales:")
print(monthly_sales)

# ── Step 4: Train Model ───────────────────────────────────
X = monthly_sales[['MonthIndex', 'Month']]
y = monthly_sales['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nMAE:  £{mean_absolute_error(y_test, y_pred):,.2f}")
print(f"RMSE: £{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

# ── Step 5: Forecast Next 6 Months ───────────────────────
last_index = monthly_sales['MonthIndex'].max()
last_month  = monthly_sales['Month'].iloc[-1]

future = []
for i in range(1, 7):
    next_month = (last_month + i - 1) % 12 + 1
    future.append({'MonthIndex': last_index + i, 'Month': next_month})

future_df    = pd.DataFrame(future)
future_preds = model.predict(future_df)

print("\nForecast for next 6 months:")
for i, pred in enumerate(future_preds, 1):
    print(f"  Month +{i}: £{pred:,.2f}")

# ── Step 6: Visualize & Save ──────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(monthly_sales['MonthIndex'], monthly_sales['Revenue'],
         label='Actual Sales', marker='o', color='blue')
plt.plot(X_test['MonthIndex'], y_pred,
         label='Model Predictions', linestyle='--', color='orange')
plt.plot(future_df['MonthIndex'], future_preds,
         label='Future Forecast', linestyle='--', color='green', marker='x')
plt.title('Monthly Sales Forecast – Online Retail')
plt.xlabel('Month Index')
plt.ylabel('Revenue (£)')
plt.legend()
plt.tight_layout()
plt.savefig(r"D:\FUTURE_ML_01\outputs\forecast.png", dpi=150)
print("\n✅ Forecast chart saved to outputs/forecast.png")
plt.show()