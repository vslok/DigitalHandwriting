import pandas as pd

# Load the CSV file
df = pd.read_csv('data/DSL-StrongPasswordData.csv')

# Initialize a new DataFrame to store the combined data
combined_df = pd.DataFrame()

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Create a dictionary to store combined data for the current row
    combined_row = {
        'subject': row['subject'],
        'sessionIndex': row['sessionIndex'],
        'rep': row['rep'],
        'password': '.tie5Roanl',
        'H': [],
        'DD': [],
        'UD': []
    }

    # Iterate over each column in the row
    for col in df.columns:
        if col.startswith('H.'):
            combined_row['H'].append(row[col])
        elif col.startswith('DD.'):
            combined_row['DD'].append(row[col])
        elif col.startswith('UD.'):
            combined_row['UD'].append(row[col])

    # Append the combined row to the new DataFrame
    combined_df = pd.concat([combined_df, pd.DataFrame([combined_row])], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('data/DSL-CombinedData.csv', index=False)

# Create and save first row for each subject
first_rows_df = combined_df.groupby('subject').first().reset_index()
first_rows_df.to_csv('data/DSL-FirstRowsData.csv', index=False)
