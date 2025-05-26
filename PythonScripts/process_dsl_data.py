import pandas as pd
import numpy as np
import json

TRAIN_ROWS_COUNT = 3

# Load the CSV file
df = pd.read_csv('data/DSL-StrongPasswordData.csv')

# Initialize a new DataFrame to store the combined data
combined_df = pd.DataFrame()

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    combined_row = {
        'Login': row['subject'],
        'sessionIndex': row['sessionIndex'],
        'rep': row['rep'],
        'Password': '.tie5Roanl',
        'H': [],
        'UD': []
    }
    for col in df.columns:
        if col.startswith('H.'):
            try:
                combined_row['H'].append(float(row[col]))
            except ValueError:
                combined_row['H'].append(0.0)
        elif col.startswith('UD.'):
            try:
                combined_row['UD'].append(float(row[col]))
            except ValueError:
                combined_row['UD'].append(0.0)
    combined_df = pd.concat([combined_df, pd.DataFrame([combined_row])], ignore_index=True)

users_df = pd.DataFrame()
all_unique_logins = combined_df['Login'].unique()

for login in all_unique_logins:
    user_data = combined_df[combined_df['Login'] == login]
    first_n_rows = user_data.iloc[:TRAIN_ROWS_COUNT]
    h_original_samples = first_n_rows['H'].tolist()
    ud_original_samples = first_n_rows['UD'].tolist()
    cleaned_h_samples = []
    for sample_vector in h_original_samples:
        if isinstance(sample_vector, list):
            cleaned_h_samples.append([0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in sample_vector])
        else:
            cleaned_h_samples.append([])
    cleaned_ud_samples = []
    for sample_vector in ud_original_samples:
        if isinstance(sample_vector, list):
            cleaned_ud_samples.append([0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in sample_vector])
        else:
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

users_df.to_csv('data/DSL-TestUsers.csv', index=False)

auth_df = pd.DataFrame()

for login in all_unique_logins:
    user_data_for_auth = combined_df[combined_df['Login'] == login].iloc[TRAIN_ROWS_COUNT:]
    num_legit_attempts_for_login = len(user_data_for_auth)

    if num_legit_attempts_for_login > 0:
        for _, row_legit in user_data_for_auth.iterrows():
            h_data = row_legit['H'] if isinstance(row_legit['H'], list) else []
            ud_data = row_legit['UD'] if isinstance(row_legit['UD'], list) else []
            h_data_json_str = json.dumps([0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in h_data])
            ud_data_json_str = json.dumps([0.0 if np.isnan(x) or np.isinf(x) else float(x) for x in ud_data])
            auth_row = {
                'Login': login,
                'H': h_data_json_str,
                'UD': ud_data_json_str,
                'IsLegalUser': True
            }
            auth_df = pd.concat([auth_df, pd.DataFrame([auth_row])], ignore_index=True)

    other_user_logins_for_impostors = [ul for ul in all_unique_logins if ul != login]

    if num_legit_attempts_for_login == 0:
        # print(f"Info: User {login} has no legitimate authentication attempts, so no impostor attempts generated for balance.")
        pass
    elif not other_user_logins_for_impostors:
        print(f"Warning: No other users available for user {login} to create impostor attempts (target: {num_legit_attempts_for_login}).")
    else:
        count_other_users = len(other_user_logins_for_impostors)

        base_samples_per_each_other_user = num_legit_attempts_for_login // count_other_users
        remainder_samples_to_distribute = num_legit_attempts_for_login % count_other_users

        total_impostors_collected_for_login = 0

        # Optional: Shuffle other_user_logins_for_impostors for fairer distribution of remainder
        # import random
        # random.Random(42).shuffle(other_user_logins_for_impostors) # For reproducible shuffle

        for i, other_login_name in enumerate(other_user_logins_for_impostors):
            if total_impostors_collected_for_login >= num_legit_attempts_for_login:
                break

            data_from_specific_other_user = combined_df[combined_df['Login'] == other_login_name]
            if data_from_specific_other_user.empty:
                continue

            samples_to_request_this_turn = base_samples_per_each_other_user
            if i < remainder_samples_to_distribute:
                samples_to_request_this_turn += 1

            num_samples_can_take_from_this_user = min(samples_to_request_this_turn, len(data_from_specific_other_user))
            num_samples_actually_needed_now = num_legit_attempts_for_login - total_impostors_collected_for_login
            num_samples_to_take = min(num_samples_can_take_from_this_user, num_samples_actually_needed_now)

            if num_samples_to_take > 0:
                impostor_samples = data_from_specific_other_user.sample(
                    n=num_samples_to_take,
                    random_state=42,
                    replace=False
                )
                for _, imp_row in impostor_samples.iterrows():
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
                    total_impostors_collected_for_login += 1

        if total_impostors_collected_for_login < num_legit_attempts_for_login:
            print(f"Warning: For user {login}, aimed for {num_legit_attempts_for_login} impostors, but only collected {total_impostors_collected_for_login} due to limited data from other users.")


auth_df.to_csv('data/DSL-AuthenticationTestData.csv', index=False)

print("Python script finished processing. DSL-TestUsers.csv now contains original samples.")
print(f"DSL-AuthenticationTestData.csv generated with impostor attempts dynamically balanced against legitimate attempts for each user.")
