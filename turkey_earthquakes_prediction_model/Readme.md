# README: Earthquake Data Analysis Pipeline

This repository contains a series of Python scripts designed to preprocess, analyze, and model earthquake data. The scripts are divided into two main categories: **classification** and **regression**. Below is a detailed explanation of each script and its purpose.

---

## 1. **Data Cleaning Scripts**

### `cleaning data 1(deletion).py`
This script is responsible for cleaning the raw earthquake dataset by removing rows with missing or invalid data.

- **Input**: Raw earthquake dataset (`turkey_earthquakes(1915-2023_may).csv`).
- **Steps**:
  1. **Drop rows with missing 'Mw' values**: The script removes rows where the 'Mw' (moment magnitude) column has null values.
  2. **Convert 'Time of occurrence' to datetime**: The script attempts to convert the 'Time of occurrence' column to a datetime format. If the conversion fails (due to incorrect formatting), those rows are dropped.
  3. **Save the cleaned dataset**: The cleaned dataset is saved as `turkey_earthquakes_cleaned.csv`.

- **Output**: A cleaned dataset ready for further analysis.

---

## 2. **Classification Models**

### `choosing classification model (1).py`
This script evaluates different classification models to predict earthquake regions based on latitude and longitude.

- **Input**: Cleaned dataset (`turkey_earthquakes_cleaned.csv`).
- **Steps**:
  1. **Preprocess the data**: The script converts the 'Date of occurrence' column into separate 'Year', 'Month', and 'Day' columns and drops the original 'Date' column.
  2. **K-means clustering**: The script uses K-means clustering to divide the geographic area into 5 distinct regions based on latitude and longitude.
  3. **Model evaluation**: The script evaluates three classification models:
     - **Random Forest**
     - **Support Vector Machine (SVM)**
     - **Neural Network (MLP)**
  4. **Metrics**: For each model, the script calculates and prints the accuracy, precision, and recall.

- **Output**: Performance metrics for each classification model.

---

### `choosing classification model (2).py`
This script is similar to the first classification script but includes an additional feature: the largest magnitude value (`xM`) among the given magnitude types (MD, ML, Mw, Ms, Mb).

- **Input**: Cleaned dataset (`turkey_earthquakes_cleaned.csv`).
- **Steps**:
  1. **Preprocess the data**: Similar to the first script, but the K-means clustering now includes the `xM` feature.
  2. **Model evaluation**: The same three models are evaluated, but with the additional `xM` feature.
  3. **Metrics**: Accuracy, precision, and recall are calculated for each model.

- **Output**: Performance metrics for each classification model with the additional `xM` feature.

---

### `choosing classification model (3).py`
This script extends the previous classification scripts by adding another feature: earthquake depth.

- **Input**: Cleaned dataset (`turkey_earthquakes_cleaned.csv`).
- **Steps**:
  1. **Preprocess the data**: Similar to the previous scripts, but the K-means clustering now includes both `xM` and `Depth` features.
  2. **Model evaluation**: The same three models are evaluated, but with the additional `Depth` feature.
  3. **Metrics**: Accuracy, precision, and recall are calculated for each model.

- **Output**: Performance metrics for each classification model with the additional `Depth` feature.

---

## 3. **Regression Models**

### `choosing regression model (1).py`
This script evaluates different regression models to predict the date (year, month, day) of earthquake occurrences.

- **Input**: Cleaned dataset (`turkey_earthquakes_cleaned.csv`).
- **Steps**:
  1. **Preprocess the data**: The script converts the 'Date of occurrence' and 'Time of occurrence' columns into separate features (Year, Month, Day, Hour, Minute, Second) and drops the original columns.
  2. **Model evaluation**: The script evaluates three regression models:
     - **Linear Regression**
     - **Decision Tree Regressor**
     - **Random Forest Regressor**
  3. **Metrics**: For each model, the script calculates and prints the Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) score.

- **Output**: Performance metrics for each regression model.

---

### `choosing regression model (2).py`
This script is similar to the first regression script but includes additional time features (Hour, Minute, Second) in the target variable.

- **Input**: Cleaned dataset (`turkey_earthquakes_cleaned.csv`).
- **Steps**:
  1. **Preprocess the data**: Similar to the first regression script, but the target variable now includes Hour, Minute, and Second.
  2. **Model evaluation**: The same three regression models are evaluated, but with the additional time features.
  3. **Metrics**: MAE, MSE, and R² are calculated for each model.

- **Output**: Performance metrics for each regression model with additional time features.

---

## Summary

- **Data Cleaning**: The `cleaning data 1(deletion).py` script prepares the raw dataset by removing rows with missing or invalid data.
- **Classification Models**: The `choosing classification model (1).py`, `(2).py`, and `(3).py` scripts evaluate classification models to predict earthquake regions, with each script adding more features (magnitude and depth) to improve accuracy.
- **Regression Models**: The `choosing regression model (1).py` and `(2).py` scripts evaluate regression models to predict the date and time of earthquake occurrences, with the second script including more granular time features.

Each script outputs performance metrics, allowing for easy comparison of different models and feature sets.
