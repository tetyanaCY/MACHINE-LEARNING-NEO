import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Зчитуємо дані
data_path = r'C:\Users\dell\MACHINE-LEARNING-NEO\final_project'
train_data = pd.read_csv(data_path + '\final_proj_data.csv')
test_data = pd.read_csv(data_path + '\final_proj_test.csv')

# Визначаємо ознаки та цільову змінну
X = train_data.drop(columns=['y'])
y = train_data['y']

# Визначаємо числові та категоріальні ознаки
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Створюємо трансформери для числових та категоріальних ознак
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Об'єднуємо трансформери
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

# Створюємо модельний пайплайн з балансуванням класів
model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE()),
    ('classifier', RandomForestClassifier())
])

# Визначаємо гіперпараметри для налаштування
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Використовуємо GridSearchCV для пошуку кращих гіперпараметрів
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)

# Навчаємо модель
grid_search.fit(X, y)

# Виводимо найкращі параметри
print(f'Найкращі параметри: {grid_search.best_params_}')

# Оцінюємо модель на крос-валідації
cv_results = cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring='balanced_accuracy')
print(f'Збалансована точність на крос-валідації: {np.mean(cv_results)}')

# Передбачення для тестового набору даних
test_predictions = grid_search.predict(test_data)

# Зберігаємо результати у потрібний формат
submission = pd.DataFrame({'index': test_data.index, 'y': test_predictions})
submission.to_csv(data_path + '\submission.csv', index=False)
