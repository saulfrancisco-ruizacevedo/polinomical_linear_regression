'''
Predicting Travel Expenses for Weekend Getaways
Description: This problem focuses on predicting the travel expenses incurred by individuals during weekend getaways. 
Understanding these expenses can help travel agencies tailor their offerings to meet customer preferences.

Features:

Distance Traveled (in miles): The distance an individual travels to their getaway destination.
Number of Participants: The total number of people accompanying the individual on the trip.
Age: The age of the individual.
Accommodation Quality: Rated from 1 to 5, where 1 is budget accommodation and 5 is luxury accommodation.
Duration of Stay (in days): The total number of days spent on the getaway.
The goal is to create a polynomial regression model to predict travel expenses in USD for weekend getaways based on these features.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# Add pandas df display options
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.options.display.float_format = '{:,.2f}'.format

# Load training data
training_data = pd.read_csv("training_data/expenses_for_weekend_vacations_dataset.csv")
# training_data = pd.read_csv("training_data/expenses_for_weekend_vacations_dataset_large.csv")


X = training_data.iloc[:, :-1].values
Y = training_data.iloc[:, -1:].values.ravel()

# Scaling features
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Transform to n^2 using polinomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_norm)

# Fit linear regression model
sgdr = SGDRegressor(max_iter=10_000)
sgdr.fit(X_poly, Y)
Y_pred = sgdr.predict(X_poly)

w = sgdr.coef_
b = sgdr.intercept_

print("Model Results and metrics")
print("=" * 40)
r2 = r2_score(Y, Y_pred)
mae = mean_absolute_error(Y, Y_pred)
print(f"R2 score: {r2 * 100:,.2}%")
print(f"Mae: {r2:,.2} USD")
print(f"Iterations: {sgdr.n_iter_}")
print(f"b: {b}")
print(f"w: {w}")




#############################################################
### Predict the travel expenses for the following dataset ###
#############################################################
new_travel_expenses = pd.DataFrame({
    'Distance Traveled (in miles)': [120, 250, 75, 300, 150],
    'Number of Participants': [2, 5, 1, 3, 4],
    'Age': [25, 34, 47, 29, 60],
    'Accommodation Quality': [3, 5, 2, 4, 5],
    'Duration of Stay (in days)': [2, 5, 1, 3, 4]
})
new_travel_expenses_X = new_travel_expenses.iloc[:, :].values
new_travel_expenses_X_norm = scaler.transform(new_travel_expenses_X)
new_travel_expenses_X_poly = poly.transform(new_travel_expenses_X_norm)
new_travel_expenses_X_predict = sgdr.predict(new_travel_expenses_X_poly)

result = pd.concat([
    new_travel_expenses,
    pd.DataFrame(data=new_travel_expenses_X_predict, columns=["Travel Expenses (USD)"])
], axis=1)


print("\n\nPredicted Travel Expenses (USD) for the given dataset")
print("=" * 40)
print(result)
