import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)  # Замените на ваш файл с данными

# Масштабирование признаков
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = data.loc[:, column_sels]
y = data['MEDV']

# Логарифмическое преобразование
y = np.log1p(y)
for col in x.columns:
    if np.abs(x[col].skew()) > 0.3:
        x[col] = np.log1p(x[col])

x = pd.DataFrame(min_max_scaler.fit_transform(x), columns=column_sels)

# Функция для оценки модели
def evaluate_model(model, x, y):
    kf = KFold(n_splits=10)
    scores = cross_val_score(model, x, y, cv=kf, scoring='neg_mean_squared_error')
    return scores

# Streamlit интерфейс
st.title('Демонстрация различных моделей регрессии')
st.write('Используйте слайдер для изменения параметра регуляризации (alpha) модели Ridge.')

# Слайдер для выбора параметра alpha для Ridge Regression
alpha = st.slider('Alpha (Ridge Regression)', min_value=0.01, max_value=100.0, step=0.01)

# Обучение и оценка модели Ridge Regression
ridge_model = linear_model.Ridge(alpha=alpha)
ridge_scores = evaluate_model(ridge_model, x, y)
st.write(f'Ridge Regression: MSE={ridge_scores.mean():.2f} (+/- {ridge_scores.std():.2f})')

# Визуализация распределения ошибок для Ridge Regression
fig_ridge, ax_ridge = plt.subplots()
sns.boxplot(data=[ridge_scores], ax=ax_ridge)
ax_ridge.set_title(f'Распределение среднеквадратичной ошибки для Ridge Regression с alpha={alpha}')
ax_ridge.set_ylabel('Отрицательная среднеквадратичная ошибка (MSE)')
st.pyplot(fig_ridge)

# Добавление дополнительных моделей
st.header('Дополнительные модели')

# Линейная регрессия
l_regression = linear_model.LinearRegression()
l_regression_scores = evaluate_model(l_regression, x, y)
st.write(f'Линейная регрессия: MSE={l_regression_scores.mean():.2f} (+/- {l_regression_scores.std():.2f})')

# Визуализация распределения ошибок для Linear Regression
fig_linear, ax_linear = plt.subplots()
sns.boxplot(data=[l_regression_scores], ax=ax_linear)
ax_linear.set_title('Распределение среднеквадратичной ошибки для Линейной регрессии')
ax_linear.set_ylabel('Отрицательная среднеквадратичная ошибка (MSE)')
st.pyplot(fig_linear)

# Ridge регрессия
l_ridge = linear_model.Ridge()
l_ridge_scores = evaluate_model(l_ridge, x, y)
st.write(f'Ridge регрессия: MSE={l_ridge_scores.mean():.2f} (+/- {l_ridge_scores.std():.2f})')

# Визуализация распределения ошибок для Ridge Regression (без задания параметра alpha)
fig_ridge2, ax_ridge2 = plt.subplots()
sns.boxplot(data=[l_ridge_scores], ax=ax_ridge2)
ax_ridge2.set_title('Распределение среднеквадратичной ошибки для Ridge регрессии (без задания alpha)')
ax_ridge2.set_ylabel('Отрицательная среднеквадратичная ошибка (MSE)')
st.pyplot(fig_ridge2)

# Полиномиальная Ridge регрессия
polynomial_ridge_model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge())
polynomial_ridge_scores = evaluate_model(polynomial_ridge_model, x, y)
st.write(f'Полиномиальная Ridge регрессия: MSE={polynomial_ridge_scores.mean():.2f} (+/- {polynomial_ridge_scores.std():.2f})')

# Визуализация распределения ошибок для Polynomial Ridge Regression
fig_polynomial_ridge, ax_polynomial_ridge = plt.subplots()
sns.boxplot(data=[polynomial_ridge_scores], ax=ax_polynomial_ridge)
ax_polynomial_ridge.set_title('Распределение среднеквадратичной ошибки для Полиномиальной Ridge регрессии')
ax_polynomial_ridge.set_ylabel('Отрицательная среднеквадратичная ошибка (MSE)')
st.pyplot(fig_polynomial_ridge)

# Gradient Boosting Regression
gradient_boosting_model = GradientBoostingRegressor(alpha=0.9, learning_rate=0.05, max_depth=2, min_samples_leaf=5, min_samples_split=2, n_estimators=100, random_state=30)
gradient_boosting_scores = evaluate_model(gradient_boosting_model, x, y)
st.write(f'Gradient Boosting Regression: MSE={gradient_boosting_scores.mean():.2f} (+/- {gradient_boosting_scores.std():.2f})')

# Визуализация распределения ошибок для Gradient Boosting Regression
fig_gradient_boosting, ax_gradient_boosting = plt.subplots()
sns.boxplot(data=[gradient_boosting_scores], ax=ax_gradient_boosting)
ax_gradient_boosting.set_title('Распределение среднеквадратичной ошибки для Gradient Boosting Regression')
ax_gradient_boosting.set_ylabel('Отрицательная среднеквадратичная ошибка (MSE)')
st.pyplot(fig_gradient_boosting)

# K-Nearest Neighbors Regression
knn_model = KNeighborsRegressor(n_neighbors=7)
knn_scores = evaluate_model(knn_model, x, y)
st.write(f'K-Nearest Neighbors Regression: MSE={knn_scores.mean():.2f} (+/- {knn_scores.std():.2f})')

# Визуализация распределения ошибок для K-Nearest Neighbors Regression
fig_knn, ax_knn = plt.subplots()
sns.boxplot(data=[knn_scores], ax=ax_knn)
ax_knn.set_title('Распределение среднеквадратичной ошибки для K-Nearest Neighbors Regression')
ax_knn.set_ylabel('Отрицательная среднеквадратичная ошибка (MSE)')
st.pyplot(fig_knn)

def evaluate_model2(model, x, y):
    scores = -1 * cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=5)
    return scores

# Масштабирование признаков
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = data.loc[:, column_sels]
y = data['MEDV']

# Логарифмическое преобразование
y = np.log1p(y)
for col in x.columns:
    if np.abs(x[col].skew()) > 0.3:
        x[col] = np.log1p(x[col])

x = pd.DataFrame(min_max_scaler.fit_transform(x), columns=column_sels)

# Определяем гиперпараметр age
age_value = st.slider("Значение параметра AGE", float(x['AGE'].min()), float(x['AGE'].max()), float(x['AGE'].mean()))

# Обновляем значение AGE
x['AGE'] = age_value

# Линейная регрессия
l_regression = linear_model.LinearRegression()
l_regression_scores = evaluate_model2(l_regression, x, y)
st.write(f'Линейная регрессия: MSE={l_regression_scores.mean():.2f} (+/- {l_regression_scores.std():.2f})')

# Визуализация распределения ошибок для Linear Regression
fig_linear, ax_linear = plt.subplots()
sns.boxplot(data=[l_regression_scores], ax=ax_linear)
ax_linear.set_title('Распределение среднеквадратичной ошибки для Линейной регрессии')
ax_linear.set_ylabel('Отрицательная среднеквадратичная ошибка (MSE)')
st.pyplot(fig_linear)


