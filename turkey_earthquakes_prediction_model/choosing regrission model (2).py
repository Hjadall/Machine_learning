
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the cleaned dataset with imputed missing values
cleaned_file_path = r'C:\Users\Huthaifa\Documents\New folder\pipeline\turkey_earthquakes_cleaned.csv'
earthquakes = pd.read_csv(cleaned_file_path)

# Preprocess the date variable
earthquakes['Date'] = pd.to_datetime(earthquakes['Date of occurrence'])
earthquakes['Year'] = earthquakes['Date'].dt.year
earthquakes['Month'] = earthquakes['Date'].dt.month
earthquakes['Day'] = earthquakes['Date'].dt.day

# Preprocess the 'Time of Occurrence' variable
earthquakes['Time of occurrence'] = pd.to_datetime(earthquakes['Time of occurrence'])
earthquakes['Hour'] = earthquakes['Time of occurrence'].dt.hour
earthquakes['Minute'] = earthquakes['Time of occurrence'].dt.minute
earthquakes['Second'] = earthquakes['Time of occurrence'].dt.second

# Drop the original 'Time of Occurrence' and 'Date' columns if needed
earthquakes.drop(['Time of occurrence', 'Date'], axis=1, inplace=True)

# Ensure all columns are in numeric format
earthquakes = pd.get_dummies(earthquakes)  # This is a simple way to handle categorical columns, you may need more sophisticated encoding methods

# Split the dataset into training and testing sets without specifying a specific year
train_set, test_set = train_test_split(earthquakes, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Training set shape:", train_set.shape)
print("Testing set shape:", test_set.shape)

# Define features and target variable
X_train = train_set.drop(['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'], axis=1)
y_train = train_set[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']]
X_test = test_set.drop(['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'], axis=1)
y_test = test_set[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']]

# List of regression models to try
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_estimators=100, random_state=42),
    # Add more models as needed
]

# Loop through models, fit, predict, and evaluate
for model in models:
    model_name = model.__class__.__name__
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    # Print evaluation metrics
    print(f"{model_name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R^2: {r2:.4f}")
