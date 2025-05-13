import pandas as pd
import numpy as np
import json

train_rows_count = 200
# Load the CSV file
df = pd.read_csv('data/DSL-StrongPasswordData.csv')

# Initialize a new DataFrame to store the combined data
combined_df = pd.DataFrame()

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Create a dictionary to store combined data for the current row
    combined_row = {
        'Login': row['subject'],  # Renamed from 'subject' to 'Login'
        'sessionIndex': row['sessionIndex'],
        'rep': row['rep'],
        'Password': '.tie5Roanl',  # Renamed from 'password' to 'Password'
        'H': [],
        'UD': []
    }

    # Iterate over each column in the row
    for col in df.columns:
        if col.startswith('H.'):
            # Ensure data is converted to float, handle non-numeric if necessary
            try:
                combined_row['H'].append(float(row[col]))
            except ValueError:
                combined_row['H'].append(0.0) # Or some other default / NaN handling
        elif col.startswith('UD.'):
            try:
                combined_row['UD'].append(float(row[col]))
            except ValueError:
                combined_row['UD'].append(0.0)

    # Append the combined row to the new DataFrame
    combined_df = pd.concat([combined_df, pd.DataFrame([combined_row])], ignore_index=True)

# Create users DataFrame matching CsvImportUser class
users_df = pd.DataFrame()

# For each unique subject/login
for login in combined_df['Login'].unique():
    user_data = combined_df[combined_df['Login'] == login]

    first_n_rows = user_data.iloc[:train_rows_count]

    h_original_samples = first_n_rows['H'].tolist()
    ud_original_samples = first_n_rows['UD'].tolist()

    # Clean each individual sample vector (list of numbers)
    cleaned_h_samples = []
    for sample_vector in h_original_samples:
        if isinstance(sample_vector, list):
            cleaned_h_samples.append([0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in sample_vector])
        else:
            # Handle cases where an expected list of numbers is not a list (e.g. if source data was bad)
            # print(f"Warning for user {login}: H sample data was not a list: {sample_vector}. Storing as empty list.")
            cleaned_h_samples.append([])

    cleaned_ud_samples = []
    for sample_vector in ud_original_samples:
        if isinstance(sample_vector, list):
            cleaned_ud_samples.append([0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in sample_vector])
        else:
            # print(f"Warning for user {login}: UD sample data was not a list: {sample_vector}. Storing as empty list.")
            cleaned_ud_samples.append([])

    h_sample_values_json_str = json.dumps(cleaned_h_samples)
    ud_sample_values_json_str = json.dumps(cleaned_ud_samples)

    user_row = {
        'Login': login,
        'Password': '.tie5Roanl',
        'HSampleValues': h_sample_values_json_str,
        'UDSampleValues': ud_sample_values_json_str
    }
    users_df = pd.concat([users_df, pd.DataFrame([user_row])], ignore_index=True)

# Save users DataFrame
users_df.to_csv('data/DSL-TestUsers.csv', index=False)

# Create authentication DataFrame matching CsvImportAuthentication class
auth_df = pd.DataFrame()

# Process 20 rows for each user (after the first N used for means)
for login in combined_df['Login'].unique():
    user_data = combined_df[combined_df['Login'] == login].iloc[train_rows_count:]

    legitimate_attempts = user_data

    if len(legitimate_attempts) > 0:
        for _, row in legitimate_attempts.iterrows():
            # Ensure row['H'] and row['UD'] are lists of numbers
            h_data = row['H'] if isinstance(row['H'], list) else []
            ud_data = row['UD'] if isinstance(row['UD'], list) else []

            # Convert lists to JSON strings for CSV if DoubleArrayConverter expects JSON string
            h_data_json_str = json.dumps([0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in h_data])
            ud_data_json_str = json.dumps([0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in ud_data])

            auth_row = {
                'Login': login,
                'H': h_data_json_str, # Store as JSON string
                'UD': ud_data_json_str, # Store as JSON string
                'IsLegalUser': True
            }
            auth_df = pd.concat([auth_df, pd.DataFrame([auth_row])], ignore_index=True)

        all_other_users_data = combined_df[combined_df['Login'] != login]
        if not all_other_users_data.empty:
            impostor_sample_count = min(len(legitimate_attempts), len(all_other_users_data)) # Match count of legit attempts for balance
            # Ensure we don't request more samples than available if replace=False
            # If replace=True (as original), then impostor_sample_count can be larger.
            # Using len(legitimate_attempts) for impostor_sample_count for balance.
            if impostor_sample_count > 0:
                # Corrected sampling:
                actual_impostor_samples_to_take = min(impostor_sample_count, len(all_other_users_data))
                other_users_data_samples = all_other_users_data.sample(n=actual_impostor_samples_to_take, replace=(actual_impostor_samples_to_take > len(all_other_users_data)), random_state=42)

                for _, imp_row in other_users_data_samples.iterrows():
                    h_imp_data = imp_row['H'] if isinstance(imp_row['H'], list) else []
                    ud_imp_data = imp_row['UD'] if isinstance(imp_row['UD'], list) else []

                    h_imp_data_json_str = json.dumps([0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in h_imp_data])
                    ud_imp_data_json_str = json.dumps([0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in ud_imp_data])

                    auth_row = {
                        'Login': login,
                        'H': h_imp_data_json_str,
                        'UD': ud_imp_data_json_str,
                        'IsLegalUser': False
                    }
                    auth_df = pd.concat([auth_df, pd.DataFrame([auth_row])], ignore_index=True)
            else:
                 print(f"Warning: Not enough legitimate attempts or impostor data for user {login} to create balanced impostor set.")

        else:
            print(f"Warning: No data available from other users to create impostor attempts for user {login}.")

# Save authentication DataFrame
auth_df.to_csv('data/DSL-AuthenticationTestData.csv', index=False)

print("Python script finished processing. DSL-TestUsers.csv now contains original samples.")
