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
            combined_row['H'].append(int(row[col] * 1000))
        elif col.startswith('DD.'):
            combined_row['DD'].append(int(row[col] * 1000))
        elif col.startswith('UD.'):
            combined_row['UD'].append(int(row[col] * 1000))

    # Append the combined row to the new DataFrame
    combined_df = pd.concat([combined_df, pd.DataFrame([combined_row])], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('data/DSL-CombinedData.csv', index=False)

# Create and save first row for each subject
first_rows_df = combined_df.groupby('subject').first().reset_index()
first_rows_df.to_csv('data/DSL-FirstRowsData.csv', index=False)

# Create test data by mixing users with different passwords
test_df = pd.DataFrame()
password_groups = df.groupby('password')
passwords = df['password'].unique()

for password1 in passwords:
    # Get users who typed password1
    users_password1 = df[df['password'] == password1]['subject'].unique()

    # Get timing data from users who typed different passwords
    other_passwords = [p for p in passwords if p != password1]
    for password2 in other_passwords:
        users_password2 = df[df['password'] == password2]['subject'].unique()

        # For each user who typed password1, get timing from a user who typed password2
        for i, login_user in enumerate(users_password1):
            timing_user = users_password2[i % len(users_password2)]

            # Get login data
            login_data = combined_df[combined_df['subject'] == login_user].iloc[0].copy()
            # Get timing data
            timing_data = combined_df[combined_df['subject'] == timing_user].iloc[0]

            # Update timing data while keeping login info
            login_data['H'] = timing_data['H']
            login_data['DD'] = timing_data['DD']
            login_data['UD'] = timing_data['UD']

            # Add to test dataset
            test_df = pd.concat([test_df, pd.DataFrame([login_data])], ignore_index=True)

# Save the test DataFrame to a new CSV file
test_df.to_csv('data/DSL-TestData.csv', index=False)
