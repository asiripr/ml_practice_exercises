import pandas as pd
from sklearn.linear_model import LinearRegression

# simple dataset
data = pd.DataFrame({
    'Size': [1500, 2000, 2500, 3000, 3500],
    'Price': [300000, 400000, 500000, 600000, 700000]
})

# split into features (X) and target (y)
X = data[['Size']]
y = data['Price']

# initialize and train the model
model = LinearRegression()
model.fit(X, y)

# make a prediction for new house
new_house = pd.DataFrame({'Size': [2200]})
prediction = model.predict(new_house)

print(f"Predicted price for a 2,200 sq ft house: ${prediction[0]:,.2f}")
