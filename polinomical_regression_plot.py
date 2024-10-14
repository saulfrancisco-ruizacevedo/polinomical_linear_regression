import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
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

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10), constrained_layout=True)
axs = axs.flatten()

for index, column in enumerate(training_data.columns[:-1]):
    X_title = column
    X_index = training_data[column].values
    
    axs[index].scatter(X_index, Y, label="Actual Income", s=70, color="orangered", edgecolor="black", alpha=0.5)
    axs[index].scatter(X_index, Y_pred, label="Predicted Income", s=70, color="blue", edgecolor="black", alpha=0.5)
    axs[index].scatter(X_new_young_adults_data[:, index], Y_new_young_adults_data_predict, color="yellow", s=70, label="Young Adult Prediction", edgecolor="black")

    axs[index].set_ylabel("Monthly Income (MXN)")
    axs[index].set_xlabel(X_title)
    axs[index].grid(linestyle="--", color="green")
    axs[index].legend()


# Visualize predictions vs actual values
axs[-1].scatter(Y, Y_pred, alpha=0.5, color="orangered")
axs[-1].plot([Y.min(), Y.max()], [Y.min(), Y.max()], "k--", lw=2, label='Linear Regression Line', color="blue")
axs[-1].set_xlabel("Actual Income (MXN)")
axs[-1].set_ylabel("Predicted Income (MXN)")
axs[-1].set_title("Actual vs Predicted Income")
axs[-1].grid(linestyle="--", color="green")
axs[-1].legend()

plt.show()
