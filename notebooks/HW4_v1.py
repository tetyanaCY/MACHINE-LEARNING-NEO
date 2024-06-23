import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# 1. Завантаження даних
california_housing = fetch_california_housing(as_frame=True)
data = california_housing.data
target = california_housing.target

# 3.1. Очистка даних від викидів
cols_to_clean = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
data[cols_to_clean] = data[cols_to_clean].apply(zscore)  # Використання zscore для визначення викидів
data = data[(np.abs(data[cols_to_clean]) < 3).all(axis=1)]  # Видалення рядків, де хоча б одна змінна є викидом

# 3.2. Видалення ознаки з високою кореляцією
correlation_matrix = data.corr()
high_corr_var=np.where(correlation_matrix>0.8)
high_corr_var=[(correlation_matrix.columns[x],correlation_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
for var_pair in high_corr_var:
    data.drop(var_pair[1], axis=1, inplace=True)

# 4. Розділення на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(data, target[data.index], test_size=0.2, random_state=42)

# 5. Нормалізація даних
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Побудова моделі лінійної регресії
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Оцінка моделі
y_pred = model.predict(X_test)
r_sq = model.score(X_train, y_train)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# 8. Вивід результатів
print(f'R2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')

