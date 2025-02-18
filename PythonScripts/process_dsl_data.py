import pandas as pd

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
            combined_row['H'].append(row[col])
        elif col.startswith('UD.'):
            combined_row['UD'].append(row[col])

    # Append the combined row to the new DataFrame
    combined_df = pd.concat([combined_df, pd.DataFrame([combined_row])], ignore_index=True)

# Create users DataFrame matching CsvImportUser class
users_df = pd.DataFrame()

# For each unique subject/login
for login in combined_df['Login'].unique():
    user_data = combined_df[combined_df['Login'] == login].iloc[:3]  # Get first 3 rows

    if len(user_data) >= 3:  # Only process if we have at least 3 rows
        user_row = {
            'Login': login,
            'Password': '.tie5Roanl',
            'FirstH': user_data.iloc[0]['H'],
            'FirstUD': user_data.iloc[0]['UD'],
            'SecondH': user_data.iloc[1]['H'],
            'SecondUD': user_data.iloc[1]['UD'],
            'ThirdH': user_data.iloc[2]['H'],
            'ThirdUD': user_data.iloc[2]['UD']
        }
        users_df = pd.concat([users_df, pd.DataFrame([user_row])], ignore_index=True)

# Save users DataFrame
users_df.to_csv('data/DSL-TestUsers.csv', index=False)

# Create authentication DataFrame matching CsvImportAuthentication class
auth_df = pd.DataFrame()

# Process remaining rows for each user (after the first 3)
for login in combined_df['Login'].unique():
    user_data = combined_df[combined_df['Login'] == login].iloc[3:]  # Get rows after first 3

    if len(user_data) > 0:
        # Add legitimate user attempts
        for _, row in user_data.iterrows():
            auth_row = {
                'Login': login,
                'H': row['H'],
                'UD': row['UD'],
                'IsLegalUser': True
            }
            auth_df = pd.concat([auth_df, pd.DataFrame([auth_row])], ignore_index=True)

        # Add impostor attempts (using data from other users)
        other_users_data = combined_df[combined_df['Login'] != login].sample(n=len(user_data))
        for _, row in other_users_data.iterrows():
            auth_row = {
                'Login': login,  # Keep the original login
                'H': row['H'],   # But use timing data from other users
                'UD': row['UD'],
                'IsLegalUser': False
            }
            auth_df = pd.concat([auth_df, pd.DataFrame([auth_row])], ignore_index=True)

# Save authentication DataFrame
auth_df.to_csv('data/DSL-AuthenticationTestData.csv', index=False)
