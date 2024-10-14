from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

# Load training data
training_data = pd.read_csv('monthly_income_training_data.csv')
# training_data = pd.read_csv('monthly_income_training_data_large.csv')

X = training_data.iloc[:, :-1].values
Y = training_data.iloc[:, -1:].values.ravel()

# Normalize data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Transform using polinomical features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_norm)

# Fit the linear regression model
sgdr = SGDRegressor(max_iter=100_000)
sgdr.fit(X_poly, Y)
Y_pred = sgdr.predict(X_poly)

w = sgdr.coef_
b = sgdr.intercept_

# New data for young adults
new_young_adults_data = pd.DataFrame({
    "Years of Education": [12, 14, 10, 16, 18, 11],
    "Weekly Working Hours": [40, 38, 30, 45, 50, 36],
    "Age": [26, 29, 22, 32, 41, 25],
    "English Proficiency": [2, 3, 1, 4, 5, 2],
    "Residence Area": [1, 0, 1, 1, 1, 0]
})

X_new_young_adults_data = new_young_adults_data.iloc[:, :].values
X_new_young_adults_data_norm = scaler.transform(X_new_young_adults_data)
X_new_young_adults_data_poly = poly.transform(X_new_young_adults_data_norm)
Y_new_young_adults_data_predict = sgdr.predict(X_new_young_adults_data_poly)

new_young_adults_data_result = pd.concat([
    new_young_adults_data,
    pd.DataFrame(Y_new_young_adults_data_predict, columns=["Predicted Salary"])
], axis=1)

print(f"Iterations: {sgdr.n_iter_} w: {w}, b: {b}\n")

print("\nTraining Data")
print("=" * 40)
print(training_data)

print("\nPredicted Salaries for Young Adults")
print("=" * 40)
print(new_young_adults_data_result)

print("\nModel Metrics")
print("=" * 40)
mae = mean_absolute_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)
print(f"mae: {mae:.2f} MXN")
print(f"R-squared (R²): {r2 * 100:,.2f}%")