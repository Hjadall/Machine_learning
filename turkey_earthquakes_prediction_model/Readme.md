# Turkey Earthquakes Prediction Model

## Introduction
This project focuses on predicting earthquake magnitudes and locations in Turkey using machine learning techniques. By analyzing historical seismic data, the model aims to identify patterns that precede significant seismic events, providing valuable insights for disaster preparedness and risk mitigation.

## Project Overview
- **Objective**: Develop a predictive model to forecast earthquake occurrences in Turkey.
- **Data Source**: Utilizes historical earthquake data from Turkey, including parameters such as date, time, latitude, longitude, depth, and magnitude.
- **Methodology**: Employs machine learning algorithms to analyze seismic data and predict future earthquakes.

## Installation
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Hjadall/Machine_learning.git
    cd Machine_learning/turkey_earthquakes_prediction_model
    ```

2. **Set Up a Virtual Environment**:
    - Using `venv`:
        ```bash
        python3 -m venv env
        source env/bin/activate  # On Windows, use `env\Scripts\activate`
        ```
    - Using `conda`:
        ```bash
        conda create --name earthquake_prediction python=3.8
        conda activate earthquake_prediction
        ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Data Preparation**:
    - Ensure the dataset (`earthquake_data.csv`) is in the `data` directory.
    - The dataset should include columns: `date`, `time`, `latitude`, `longitude`, `depth`, `magnitude`.

2. **Model Training**:
    - Run the training script:
        ```bash
        python train_model.py
        ```
    - This script will preprocess the data, train the model, and save the trained model to the `models` directory.

3. **Making Predictions**:
    - Use the prediction script:
        ```bash
        python predict.py --input data/new_data.csv --output predictions.csv
        ```
    - Replace `data/new_data.csv` with your input file containing new seismic data.

## Features
- **Data Visualization**: Includes scripts to visualize seismic data trends and patterns.
- **Model Evaluation**: Provides tools to assess model performance using metrics like RMSE and R².
- **Geospatial Mapping**: Plots earthquake occurrences on a map for spatial analysis.

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Description of changes"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by various earthquake prediction projects and research studies focusing on Turkey.
- Utilizes data from reputable seismic databases and incorporates methodologies from recent scientific publications.
