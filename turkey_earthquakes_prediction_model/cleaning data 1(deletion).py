import pandas as pd

# Load the dataset
file_path = r'C:\Users\Huthaifa\Documents\New folder\pipeline\turkey_earthquakes(1915-2023_may).csv'
earthquakes = pd.read_csv(file_path)

# Drop rows where 'Mw' is null
earthquakes.dropna(subset=['Mw'], inplace=True)

# Convert 'Time of occurrence' to datetime format with the correct format
earthquakes['Time of occurrence'] = pd.to_datetime(earthquakes['Time of occurrence'], format='%M:%S.%f', errors='coerce')

# Drop rows with NaN values in 'Time of occurrence'
earthquakes.dropna(subset=['Time of occurrence'], inplace=True)

# Save the cleaned dataset
cleaned_file_path = r'C:\Users\Huthaifa\Documents\New folder\pipeline\turkey_earthquakes_cleaned.csv'
earthquakes.to_csv(cleaned_file_path, index=False)
