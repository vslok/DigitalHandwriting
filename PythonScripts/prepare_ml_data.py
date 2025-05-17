import pandas as pd
import numpy as np
import json

def calculate_ngraph_features(hold_times, between_times, n):
    """Calculate n-graph features for a single sequence"""
    if n == 1:
        return {
            'H': hold_times,
            'UD': between_times
        }

    if len(hold_times) < n:
        return None

    n_graph_between_keys_down = []
    n_graph_between_keys_up = []
    n_graph_hold = []
    n_graph_between_keys = []

    for i in range(len(hold_times) - n + 1):
        # Calculate DD (Down-Down)
        dd_values = []
        for j in range(n - 1):
            dd_values.append(hold_times[i + j])
            dd_values.append(between_times[i + j])
        n_graph_between_keys_down.append(sum(dd_values))

        # Calculate UU (Up-Up)
        uu_values = []
        for j in range(n - 1):
            uu_values.append(between_times[i + j])
            uu_values.append(hold_times[i + j + 1])
        n_graph_between_keys_up.append(sum(uu_values))

        # Calculate H (Hold)
        h_values = hold_times[i:i + n]
        n_graph_hold.append(sum(h_values))

        # Calculate UD (Up-Down)
        ud_values = between_times[i:i + n - 1]
        n_graph_between_keys.append(sum(ud_values))

    return {
        'H': n_graph_hold,
        'DD': n_graph_between_keys_down,
        'UU': n_graph_between_keys_up,
        'UD': n_graph_between_keys
    }

def prepare_ml_data():
    # Load the original CSV file
    df = pd.read_csv('data/DSL-StrongPasswordData.csv')

    # Define target passphrase length
    TARGET_LENGTH = 20

    # Initialize DataFrames for each n-graph level
    train_dfs = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
    train_val_dfs = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
    test_dfs = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}

    # Get unique users
    unique_users = df['subject'].unique()

    def pad_sequence(values, target_length):
        """Pad or truncate sequence to target length"""
        if len(values) < target_length:
            return values + [0.0] * (target_length - len(values))
        elif len(values) > target_length:
            return values[:target_length]
        return values

    # Process each user
    for user in unique_users:
        # Get user's own data
        user_data = df[df['subject'] == user]

        # Split user's data into train (240), train_validation (60), and test (100)
        user_train = user_data.sample(n=240, random_state=42)
        remaining_data = user_data.drop(user_train.index)
        user_train_val = remaining_data.sample(n=60, random_state=42)
        user_test = remaining_data.drop(user_train_val.index)

        # Process each dataset (train, train_val, test)
        for dataset, data in [('train', user_train), ('train_val', user_train_val), ('test', user_test)]:
            # Process legal entries
            for index, row in data.iterrows():
                # Collect H and UD values
                h_values = []
                ud_values = []
                for col in df.columns:
                    if col.startswith('H.'):
                        try:
                            h_values.append(float(row[col]))
                        except ValueError:
                            h_values.append(0.0)
                    elif col.startswith('UD.'):
                        try:
                            ud_values.append(float(row[col]))
                        except ValueError:
                            ud_values.append(0.0)

                # Calculate features for each n-graph level
                for n in [1, 2, 3]:
                    n_graph_features = calculate_ngraph_features(h_values, ud_values, n)
                    if n_graph_features is None:
                        continue

                    combined_row = {
                        'Login': row['subject'],
                        'IsLegalUser': 1
                    }

                    # Store features as space-separated strings
                    for feature_type, values in n_graph_features.items():
                        padded_values = pad_sequence(values, TARGET_LENGTH)
                        combined_row[feature_type] = ' '.join(map(str, padded_values))

                    if dataset == 'train':
                        train_dfs[n] = pd.concat([train_dfs[n], pd.DataFrame([combined_row])], ignore_index=True)
                    elif dataset == 'train_val':
                        train_val_dfs[n] = pd.concat([train_val_dfs[n], pd.DataFrame([combined_row])], ignore_index=True)
                    else:
                        test_dfs[n] = pd.concat([test_dfs[n], pd.DataFrame([combined_row])], ignore_index=True)

            # Create illegal user entries
            other_users_data = df[df['subject'] != user]
            n_samples = 240 if dataset == 'train' else (60 if dataset == 'train_val' else 100)

            if len(other_users_data) >= n_samples:
                illegal_data = other_users_data.sample(n=n_samples, random_state=42)
            else:
                illegal_data = other_users_data.sample(n=n_samples, replace=True, random_state=42)

            # Process illegal entries
            for index, row in illegal_data.iterrows():
                # Collect H and UD values
                h_values = []
                ud_values = []
                for col in df.columns:
                    if col.startswith('H.'):
                        try:
                            h_values.append(float(row[col]))
                        except ValueError:
                            h_values.append(0.0)
                    elif col.startswith('UD.'):
                        try:
                            ud_values.append(float(row[col]))
                        except ValueError:
                            ud_values.append(0.0)

                # Calculate features for each n-graph level
                for n in [1, 2, 3]:
                    n_graph_features = calculate_ngraph_features(h_values, ud_values, n)
                    if n_graph_features is None:
                        continue

                    combined_row = {
                        'Login': user,
                        'IsLegalUser': 0
                    }

                    # Store features as space-separated strings
                    for feature_type, values in n_graph_features.items():
                        padded_values = pad_sequence(values, TARGET_LENGTH)
                        combined_row[feature_type] = ' '.join(map(str, padded_values))

                    if dataset == 'train':
                        train_dfs[n] = pd.concat([train_dfs[n], pd.DataFrame([combined_row])], ignore_index=True)
                    elif dataset == 'train_val':
                        train_val_dfs[n] = pd.concat([train_val_dfs[n], pd.DataFrame([combined_row])], ignore_index=True)
                    else:
                        test_dfs[n] = pd.concat([test_dfs[n], pd.DataFrame([combined_row])], ignore_index=True)

    # Shuffle and save datasets for each n-graph level
    for n in [1, 2, 3]:
        train_dfs[n] = train_dfs[n].sample(frac=1, random_state=42).reset_index(drop=True)
        train_val_dfs[n] = train_val_dfs[n].sample(frac=1, random_state=42).reset_index(drop=True)
        test_dfs[n] = test_dfs[n].sample(frac=1, random_state=42).reset_index(drop=True)

        # Save to CSV files
        train_dfs[n].to_csv(f'data/ML_KeystrokeData_train_{n}graph.csv', index=False)
        train_val_dfs[n].to_csv(f'data/ML_KeystrokeData_train_val_{n}graph.csv', index=False)
        test_dfs[n].to_csv(f'data/ML_KeystrokeData_test_{n}graph.csv', index=False)

        print(f"\n{n}-Graph Data:")
        print(f"Training Data:")
        print(f"Total rows: {len(train_dfs[n])}")
        print(f"Legal users (1): {len(train_dfs[n][train_dfs[n]['IsLegalUser'] == 1])}")
        print(f"Illegal users (0): {len(train_dfs[n][train_dfs[n]['IsLegalUser'] == 0])}")

        print(f"\nTraining Validation Data:")
        print(f"Total rows: {len(train_val_dfs[n])}")
        print(f"Legal users (1): {len(train_val_dfs[n][train_val_dfs[n]['IsLegalUser'] == 1])}")
        print(f"Illegal users (0): {len(train_val_dfs[n][train_val_dfs[n]['IsLegalUser'] == 0])}")

        print(f"\nTest Data:")
        print(f"Total rows: {len(test_dfs[n])}")
        print(f"Legal users (1): {len(test_dfs[n][test_dfs[n]['IsLegalUser'] == 1])}")
        print(f"Illegal users (0): {len(test_dfs[n][test_dfs[n]['IsLegalUser'] == 0])}")

        print(f"\nFeature length: {TARGET_LENGTH}")

if __name__ == "__main__":
    prepare_ml_data()
