import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings

# Ignore convergence warnings during training
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load the cleaned dataset with imputed missing values
cleaned_file_path = r'C:\Users\Huthaifa\Documents\New folder\pipeline\turkey_earthquakes_cleaned.csv'
earthquakes = pd.read_csv(cleaned_file_path)

# Preprocess the date variable
earthquakes['Date'] = pd.to_datetime(earthquakes['Date of occurrence'])
earthquakes['Year'] = earthquakes['Date'].dt.year
earthquakes['Month'] = earthquakes['Date'].dt.month
earthquakes['Day'] = earthquakes['Date'].dt.day

# Drop the original 'Time of occurrence' and 'Date' columns if needed
earthquakes.drop(['Date'], axis=1, inplace=True)

# Ensure all columns are in numeric format
earthquakes = pd.get_dummies(earthquakes)

# Use K-means clustering to divide the geographic area into distinct regions
num_clusters = 5  # Adjust this based on your specific requirements
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit(earthquakes[['Latitude', 'Longitude']])
earthquakes['Region'] = kmeans.labels_

# Implement and evaluate multiple classification models
classification_models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Neural Network (MLP)': MLPClassifier(max_iter=500, solver='adam', random_state=42)
}

for model_name, model in classification_models.items():
    # Prepare data for classification
    X_classification = earthquakes[['Latitude', 'Longitude']]
    y_classification = earthquakes['Region']

    # Split the dataset into training and testing sets
    train_set_class, test_set_class = train_test_split(earthquakes, test_size=0.2, random_state=42)

    # Train the classification model
    model.fit(train_set_class[['Latitude', 'Longitude']], train_set_class['Region'])

    # Make predictions on the test set
    predictions = model.predict(test_set_class[['Latitude', 'Longitude']])

    # Evaluate the classification model
    accuracy = accuracy_score(test_set_class['Region'], predictions)
    precision = precision_score(test_set_class['Region'], predictions, average='weighted')
    recall = recall_score(test_set_class['Region'], predictions, average='weighted')

    # Display the evaluation metrics for each model
    print(f"\n{model_name} Classification Results:")
    print("Classification Accuracy:", accuracy)
    print("Classification Precision:", precision)
    print("Classification Recall:", recall)
