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

# Create test data
test_df = pd.DataFrame()
subjects = combined_df['subject'].unique()

# For each subject
for subject in subjects:
    # Get the subject's login data as template
    login_data = combined_df[combined_df['subject'] == subject].iloc[0]

    # Create base DataFrame for this subject with 100 rows
    subject_rows = pd.DataFrame({
        'subject': [subject] * 100,
        'sessionIndex': [i // 20 + 1 for i in range(100)],  # 5 sessions
        'rep': [i % 20 + 1 for i in range(100)],            # 20 reps per session
        'password': [login_data['password']] * 100
    })

    # Get timing data from other subjects
    other_subjects_data = combined_df[combined_df['subject'] != subject].sample(n=100, replace=True)

    # Add timing data from other subjects
    subject_rows['H'] = other_subjects_data['H'].values
    subject_rows['DD'] = other_subjects_data['DD'].values
    subject_rows['UD'] = other_subjects_data['UD'].values

    # Add to test dataset
    test_df = pd.concat([test_df, subject_rows], ignore_index=True)

# Save the test DataFrame to a new CSV file
test_df.to_csv('data/DSL-TestData.csv', index=False)
