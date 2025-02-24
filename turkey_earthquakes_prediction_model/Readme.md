# Earthquake Prediction and Location Classification

## Project Overview

This project aims to predict the time and location of earthquakes in Turkey using machine learning models. The dataset spans from 1915 to 2023 and includes variables such as date of occurrence, latitude, longitude, depth, and magnitude values (MD, ML, Mw, Ms, Mb). The project employs both regression and classification models to predict the time of the next earthquake and classify its location, respectively.

## Key Objectives

- **Regression Models**: Predict the time of the next earthquake.
- **Classification Models**: Classify the location of the next earthquake based on historical data.

## Methodology

### Data Cleaning

Two data cleaning approaches were explored:
1. **Deletion of rows with missing values**.
2. **Mean imputation** - This approach proved more effective, enhancing model performance.

**Special Attention**:
- The 'Time of Occurrence' column was converted to datetime format using the format `'%M:%S.%f'` to handle minute, second, and microsecond components.
- The 'Date of Occurrence' column was preprocessed to extract relevant time-related features (year, month, day).
- The 'Time of Occurrence' column was preprocessed to extract relevant time-related features (hour, minute, second).

## Model Performance

### Regression Models

#### First Model (Time of Occurrence and Date of Occurrence columns)

| Model                  | MAE    | MSE     | R²     |
|------------------------|--------|---------|--------|
| Linear Regression      | 6.7370 | 157.3524| 0.3731 |
| Decision Tree Regressor| 8.1055 | 219.6095| 0.0263 |
| Random Forest Regressor| 6.3108 | 119.0472| 0.4661 |

#### Second Model (Date of Occurrence Only)

| Model                  | MAE    | MSE     | R²     |
|------------------------|--------|---------|--------|
| Linear Regression      | 1.9709 | 13.1876 | 0.7277 |
| Decision Tree Regressor| 1.7364 | 22.2929 | 0.6297 |
| Random Forest Regressor| 1.7014 | 14.1258 | 0.7622 |

**Analysis**:
- The second model, using only the date of occurrence and employing the Random Forest Regressor, outperforms the first model based on all metrics.

### Classification Models

#### First Model (Latitude and Longitude columns)

| Model                          | Accuracy | Precision | Recall  |
|--------------------------------|----------|-----------|---------|
| Random Forest Classification   | 0.9949   | 0.99497   | 0.99495 |
| Support Vector Machine (SVM)   | 0.9877   | 0.98814   | 0.98773 |
| Neural Network (MLP)           | 0.9618   | 0.96225   | 0.96176 |

#### Second Model (Latitude, Longitude, xM columns)

| Model                          | Accuracy | Precision | Recall  |
|--------------------------------|----------|-----------|---------|
| Random Forest Classification   | 0.9928   | 0.99281   | 0.99278 |
| Support Vector Machine (SVM)   | 0.9848   | 0.98523   | 0.98485 |
| Neural Network (MLP)           | 0.9646   | 0.96551   | 0.96465 |

#### Third Model (Latitude, Longitude, xM, Depth columns)

| Model                          | Accuracy | Precision | Recall  |
|--------------------------------|----------|-----------|---------|
| Random Forest Classification   | 0.9965   | 0.99648   | 0.9965  |
| Support Vector Machine (SVM)   | 0.9906   | 0.99069   | 0.9906  |
| Neural Network (MLP)           | 0.9711   | 0.97163   | 0.9711  |

**Analysis**:
- The Random Forest classification model in the third configuration (including Latitude, Longitude, xM, and Depth) exhibits exceptional accuracy, precision, and recall.
- Across all models, Random Forest consistently performs well with high accuracy, precision, and recall.
- Adding 'xM' as a feature does not significantly impact performance compared to the first model without 'xM'.

## Combined Model Performance

- **Combined Accuracy (Time)**: 0.9963
- **Combined Accuracy (Location)**: 0.9965

The combined model, considering both time and location aspects, demonstrates exceptional accuracy, indicating the robustness and effectiveness of the machine learning models in providing comprehensive predictions for both temporal and spatial aspects of earthquake occurrences.

## Visualizations

Visualizations were generated to compare actual vs. predicted values for both regression and classification models.

## Model Selection Justification

The **Random Forest model** consistently demonstrated superior performance in both regression and classification tasks. Its ability to handle complex relationships within the data, mitigate overfitting, and provide high accuracy, precision, and recall makes it the optimal choice for this project.

## Conclusion

The Random Forest model, particularly in the first classification model using only Latitude and Longitude, proves to be the optimal choice for predicting earthquake locations. The combination of regression and classification models enhances the overall predictive capabilities of the system. Further refinement and exploration of features may lead to even more accurate predictions in future iterations of the model.
