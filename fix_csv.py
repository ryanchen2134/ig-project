import os
import pandas as pd

# Path to your original CSV file
csv_file = './csv files/citypop.csv'
# Path to save the filtered CSV file
filtered_csv_file = './filtered/new_city_pop.csv'

# Load the CSV
df = pd.read_csv(csv_file)

# List to store valid rows
valid_rows = []

# Loop through each row in the CSV and check if the image exists
for index, row in df.iterrows():
    image_path = row['image_path']
    
    # Check if the file exists
    if os.path.exists(image_path):
        valid_rows.append(row)
    else:
        print(f"Image not found: {image_path}")

# Create a new DataFrame with the valid rows
filtered_df = pd.DataFrame(valid_rows)

# Save the filtered CSV
filtered_df.to_csv(filtered_csv_file, index=False)

print(f"Filtered CSV saved as: {filtered_csv_file}")
