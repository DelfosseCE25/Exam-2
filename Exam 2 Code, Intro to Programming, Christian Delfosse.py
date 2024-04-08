# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# File path to the Excel spreadsheet
file_path = r'C:\Users\DelfosseCE25\Downloads\Restaurant Revenue.xlsx'

# Reading data from Excel file
data = pd.read_excel(file_path)

# Displaying the first few rows of the dataset to understand its structure
print(data.head())

# Selecting features (independent variables) and target (dependent variable)
X = data[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = data['Monthly_Revenue']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Training the model on the training data
model.fit(X_train, y_train)

# Predicting monthly revenue using the test set
y_pred = model.predict(X_test)

# Calculating Mean Squared Error (MSE) to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Providing an example prediction using hypothetical data
example_data = [[100, 20, 5000, 30, 100, 4]]  # Example values for features
example_prediction = model.predict(example_data)
print("Example Prediction for Monthly Revenue:", example_prediction[0])

print("Go Bucks!") 
