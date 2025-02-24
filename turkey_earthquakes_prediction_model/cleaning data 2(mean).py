import pandas as pd

# Load the dataset
file_path = r'C:\Users\Huthaifa\Documents\New folder\pipeline\turkey_earthquakes(1915-2023_may).csv'
earthquakes = pd.read_csv(file_path)

# Convert 'Time of occurrence' to datetime format with the correct format
earthquakes['Time of occurrence'] = pd.to_datetime(earthquakes['Time of occurrence'], format='%M:%S.%f', errors='coerce')

# Check for rows with missing datetime values
missing_time_rows = earthquakes[earthquakes['Time of occurrence'].isnull()]
print(f"Number of rows with missing 'Time of occurrence': {len(missing_time_rows)}")

# Impute missing values in 'Mw' with the mean value
earthquakes['Mw'].fillna(earthquakes['Mw'].mean(), inplace=True)

# Impute missing values in other columns or apply additional preprocessing as needed
# Example: earthquakes['OtherColumn'].fillna(earthquakes['OtherColumn'].mean(), inplace=True)

# Save the cleaned dataset
cleaned_file_path = r'C:\Users\Huthaifa\Documents\New folder\pipeline\turkey_earthquakes_cleaned_imputedd.csv'
earthquakes.to_csv(cleaned_file_path, index=False)
