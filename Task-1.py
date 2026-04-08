import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# =========================
# STEP 1: Load Dataset
# =========================
df = pd.read_csv(r"C:\Users\saibh\OneDrive\Desktop\archive\Sample - Superstore.csv", encoding='latin1')


print(df)

# =========================
# STEP 2: Understand Data
# =========================
print("\nShape:", df.shape)
print("\nColumns:", df.columns)
print("\nInfo:")
print(df.info())

# =========================
# STEP 3: Data Cleaning
# =========================
print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

# =========================
# STEP 4: Convert Date
# =========================
df['Order Date'] = pd.to_datetime(df['Order Date'])

# =========================
# STEP 5: Feature Engineering
# =========================
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day

# =========================
# STEP 6: Aggregate Sales
# =========================
sales_data = df.groupby('Order Date')['Sales'].sum().reset_index()

# =========================
# STEP 7: Visualize Trend
# =========================
plt.figure()
plt.plot(sales_data['Order Date'], sales_data['Sales'])
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# =========================
# STEP 8: Train Model
# =========================
sales_data['Date_num'] = sales_data['Order Date'].map(pd.Timestamp.toordinal)

X = sales_data[['Date_num']]
y = sales_data['Sales']

model = LinearRegression()
model.fit(X, y)

# =========================
# STEP 9: Forecast Future
# =========================
future_dates = pd.date_range(start=sales_data['Order Date'].max(), periods=30)

future_df = pd.DataFrame({'Order Date': future_dates})
future_df['Date_num'] = future_df['Order Date'].map(pd.Timestamp.toordinal)

predictions = model.predict(future_df[['Date_num']])

# =========================
# STEP 10: Visualize Forecast
# =========================
plt.figure()
plt.plot(sales_data['Order Date'], sales_data['Sales'], label='Actual')
plt.plot(future_df['Order Date'], predictions, label='Forecast')
plt.legend()
plt.title("Sales Forecast")
plt.show()

# =========================
# STEP 11: Evaluate Model
# =========================
pred_train = model.predict(X)
mae = mean_absolute_error(y, pred_train)

print("\nModel MAE:", mae)











