import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans

# Load the cleaned dataset with imputed missing values
cleaned_file_path = r'C:\Users\Huthaifa\Documents\New folder\pipeline\turkey_earthquakes_cleaned_imputedd.csv'
earthquakes = pd.read_csv(cleaned_file_path)

# Preprocess the date variable
earthquakes['Date'] = pd.to_datetime(earthquakes['Date of occurrence'])
earthquakes['Year'] = earthquakes['Date'].dt.year
earthquakes['Month'] = earthquakes['Date'].dt.month
earthquakes['Day'] = earthquakes['Date'].dt.day

# Preprocess the 'Time of Occurrence' variable for regression
earthquakes['Time of occurrence'] = pd.to_datetime(earthquakes['Time of occurrence'])
earthquakes['Hour'] = earthquakes['Time of occurrence'].dt.hour
earthquakes['Minute'] = earthquakes['Time of occurrence'].dt.minute
earthquakes['Second'] = earthquakes['Time of occurrence'].dt.second

# Drop the original 'Time of Occurrence' and 'Date' columns if needed for regression
earthquakes.drop(['Time of occurrence', 'Date'], axis=1, inplace=True)

# Ensure all columns are in numeric format
earthquakes = pd.get_dummies(earthquakes)

# Use K-means clustering to divide the geographic area into distinct regions
num_clusters = 5  # Adjust this based on your specific requirements
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(earthquakes[['Latitude', 'Longitude','xM            The largest value among the given magnitude (MD, ML, Mw, Ms, Mb) values.','Depth']])
earthquakes['Region'] = kmeans.labels_

# Split the dataset into training and testing sets without specifying a specific year
train_set, test_set = train_test_split(earthquakes, test_size=0.2, random_state=42)

# Regression Features
regression_features = ['Year', 'Month', 'Day']
X_regression = train_set[regression_features]
y_regression = train_set[['Year', 'Month', 'Day']]  # Corrected target variable
X_test_regression = test_set[regression_features]
y_test_regression = test_set[['Year', 'Month', 'Day']]  # Corrected target variable

# Classification Features
classification_features = ['Latitude', 'Longitude','xM            The largest value among the given magnitude (MD, ML, Mw, Ms, Mb) values.','Depth']
X_classification = train_set[classification_features]
y_classification = train_set['Region']
X_test_classification = test_set[classification_features]
y_test_classification = test_set['Region']

# Train the Random Forest Regressor for time prediction
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(X_regression, y_regression)

# Train the Random Forest Classifier for location classification
random_forest_classifier = RandomForestClassifier(random_state=42)
random_forest_classifier.fit(X_classification, y_classification)

# Make predictions on the test set for both time and location
time_predictions = random_forest_regressor.predict(X_test_regression)
location_predictions = random_forest_classifier.predict(X_test_classification)


# Assess the combined model's overall performance
# You can use relevant metrics based on your project requirements
combined_accuracy_time = r2_score(y_test_regression, time_predictions)
combined_accuracy_location = accuracy_score(y_test_classification, location_predictions)

print("\nCombined Model Performance:")
print("Combined Accuracy (Time):", combined_accuracy_time)
print("Combined Accuracy (Location):", combined_accuracy_location)

import matplotlib.pyplot as plt

# Visualize the actual vs. predicted Year, Month, and Day in a 2D scatter plot
plt.figure(figsize=(12, 6))

# Actual vs. Predicted Year
plt.scatter(y_test_regression['Year'], X_test_regression['Month']*100 + X_test_regression['Day'], label='Actual', color='blue')
plt.scatter(time_predictions[:, 0], X_test_regression['Month']*100 + X_test_regression['Day'], label='Predicted', color='red')  # Assuming time_predictions is a 2D array

plt.title('Actual vs. Predicted Year')
plt.xlabel('Year')
plt.ylabel('Month and Day')
plt.legend()

plt.show()



import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Classification Map for Location Predictions
gdf_actual = gpd.GeoDataFrame(test_set, geometry=gpd.points_from_xy(test_set['Longitude'], test_set['Latitude']))
gdf_predicted = gpd.GeoDataFrame(test_set, geometry=gpd.points_from_xy(test_set['Longitude'], test_set['Latitude']))
gdf_actual['Region'] = gdf_actual['Region'].astype(str)
gdf_predicted['Region'] = gdf_predicted['Region'].astype(str)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
gdf_actual.plot(ax=ax1, marker='o', color='blue', markersize=10, label='Actual Region')
gdf_predicted.plot(ax=ax2, marker='o', color='red', markersize=10, label='Predicted Region')