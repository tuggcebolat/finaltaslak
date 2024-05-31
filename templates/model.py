import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Veriyi oku
df = pd.read_csv("templates/company_financials_log_transformed.csv")
print(df.head())

# Logaritmik olmayan sütunları silme
df = df[['month', 'log_revenue', 'log_net_income', 'log_cost_of_goods_sold',
                  'log_gross_profit', 'log_operating_expenses', 'log_operating_income']]

# Dağılım Analizi
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='log_revenue', bins=20, kde=True)
plt.title('Distribution of Log Revenue')
plt.xlabel('Log Revenue')
plt.ylabel('Frequency')
plt.show()

# Korelasyon Analizi
correlation_matrix = df[['log_revenue', 'log_net_income', 'log_cost_of_goods_sold', 'log_gross_profit', 'log_operating_expenses', 'log_operating_income']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Log-Transformed Variables')
plt.show()

# Regresyon Analizi
sns.pairplot(df, x_vars=['log_cost_of_goods_sold', 'log_gross_profit'], y_vars='log_revenue', kind='scatter', height=5, aspect=1)
plt.title('Scatter plot of Log Revenue vs Log Cost of Goods Sold and Log Gross Profit')
plt.xlabel('Log Cost of Goods Sold and Log Gross Profit')
plt.ylabel('Log Revenue')
plt.show()

# Boxplot ile aykırı değerleri göster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['log_revenue', 'log_net_income', 'log_cost_of_goods_sold', 'log_gross_profit', 'log_operating_expenses', 'log_operating_income']], orient="h", palette="Set2")
plt.title('Boxplot of Log-Transformed Variables')
plt.xlabel('Log Value')
plt.show()

# Aykırı değerler için threshold belirle
threshold = 1.5

# Aykırı değerleri hesapla
Q1 = df['log_operating_expenses'].quantile(0.25)
Q3 = df['log_operating_expenses'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - threshold * IQR
upper_bound = Q3 + threshold * IQR

# Aykırı değerleri filtrele
outliers = df[(df['log_operating_expenses'] < lower_bound) | (df['log_operating_expenses'] > upper_bound)]

# Boxplot ile aykırı değerleri göster
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['log_operating_expenses'], orient="h", palette="Set2")
plt.scatter(outliers['log_operating_expenses'], [0] * len(outliers), color='r', label='Outliers')
plt.title('Boxplot of Log Operating Expenses with Outliers')
plt.xlabel('Log Operating Expenses')
plt.legend()
plt.show()

# Aykırı değerleri filtrele
outliers_index = df[(df['log_operating_expenses'] < lower_bound) | (df['log_operating_expenses'] > upper_bound)].index

# Aykırı değerleri veri setinden çıkar
df_cleaned = df.drop(outliers_index)
df_cleaned.shape

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak veri setini ayırma
X = df_cleaned[['log_revenue', 'log_cost_of_goods_sold', 'log_operating_expenses']]
y = df_cleaned['log_net_income']

# Eğitim ve test kümelerine veri setini böleme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge regresyon modelini oluşturma ve eğitme
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Eğitim kümesi üzerinde modelin performansını değerlendirme
train_pred = model.predict(X_train)
train_rmse = mean_squared_error(y_train, train_pred, squared=False)
print("Eğitim Kümesi RMSE:", train_rmse)

# Test kümesi üzerinde modelin performansını değerlendirme
test_pred = model.predict(X_test)
test_rmse = mean_squared_error(y_test, test_pred, squared=False)
test_r2 = r2_score(y_test, test_pred)
print("Test Kümesi RMSE:", test_rmse)
print("Test Kümesi R²:", test_r2)

# Gelecekteki performansı tahmin etme
future_data = pd.DataFrame({
    'log_revenue': [np.log(250000)],  # Örneğin, gelecekteki gelir tahmini (logaritmik formatta)
    'log_cost_of_goods_sold': [np.log(150000)],  # Gelecekteki mal satış maliyeti tahmini (logaritmik formatta)
    'log_operating_expenses': [np.log(100000)]  # Gelecekteki işletme giderleri tahmini (logaritmik formatta)
})

future_net_income = model.predict(future_data)
print("Gelecekteki Net Gelir Tahmini:", future_net_income)

# Modeli pickle dosyasına kaydet
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model başarıyla kaydedildi.")

# Modeli pickle dosyasından yükle
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Modeli test et
loaded_model_test_pred = loaded_model.predict(X_test)
loaded_model_test_rmse = mean_squared_error(y_test, loaded_model_test_pred, squared=False)
loaded_model_test_r2 = r2_score(y_test, loaded_model_test_pred)

print("Yüklenen Modelin Test Kümesi RMSE:", loaded_model_test_rmse)
print("Yüklenen Modelin Test Kümesi R²:", loaded_model_test_r2)
