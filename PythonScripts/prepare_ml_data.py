import pandas as pd
import numpy as np
import json

def prepare_ml_data():
    # Load the original CSV file
    df = pd.read_csv('data/DSL-StrongPasswordData.csv')

    # Define target passphrase length
    TARGET_LENGTH = 20

    # Initialize DataFrames to store the training, train validation, and test data
    train_df = pd.DataFrame()
    train_val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Get unique users
    unique_users = df['subject'].unique()

    def pad_sequence(values, target_length):
        """Pad or truncate sequence to target length"""
        if len(values) < target_length:
            # Pad with zeros if sequence is too short
            return values + [0.0] * (target_length - len(values))
        elif len(values) > target_length:
            # Truncate if sequence is too long
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

        # Process training data (legal entries)
        for index, row in user_train.iterrows():
            combined_row = {
                'Login': row['subject'],
                'IsLegalUser': 1
            }

            # Collect H values
            h_values = []
            for col in df.columns:
                if col.startswith('H.'):
                    try:
                        h_values.append(float(row[col]))
                    except ValueError:
                        h_values.append(0.0)

            # Collect UD values
            ud_values = []
            for col in df.columns:
                if col.startswith('UD.'):
                    try:
                        ud_values.append(float(row[col]))
                    except ValueError:
                        ud_values.append(0.0)

            # Pad sequences to target length
            h_values = pad_sequence(h_values, TARGET_LENGTH)
            ud_values = pad_sequence(ud_values, TARGET_LENGTH)

            # Store as space-separated string of numbers
            combined_row['H'] = ' '.join(map(str, h_values))
            combined_row['UD'] = ' '.join(map(str, ud_values))

            train_df = pd.concat([train_df, pd.DataFrame([combined_row])], ignore_index=True)

        # Process train validation data (legal entries)
        for index, row in user_train_val.iterrows():
            combined_row = {
                'Login': row['subject'],
                'IsLegalUser': 1
            }

            # Collect H values
            h_values = []
            for col in df.columns:
                if col.startswith('H.'):
                    try:
                        h_values.append(float(row[col]))
                    except ValueError:
                        h_values.append(0.0)

            # Collect UD values
            ud_values = []
            for col in df.columns:
                if col.startswith('UD.'):
                    try:
                        ud_values.append(float(row[col]))
                    except ValueError:
                        ud_values.append(0.0)

            # Pad sequences to target length
            h_values = pad_sequence(h_values, TARGET_LENGTH)
            ud_values = pad_sequence(ud_values, TARGET_LENGTH)

            # Store as space-separated string of numbers
            combined_row['H'] = ' '.join(map(str, h_values))
            combined_row['UD'] = ' '.join(map(str, ud_values))

            train_val_df = pd.concat([train_val_df, pd.DataFrame([combined_row])], ignore_index=True)

        # Process test data (legal entries)
        for index, row in user_test.iterrows():
            combined_row = {
                'Login': row['subject'],
                'IsLegalUser': 1
            }

            # Collect H values
            h_values = []
            for col in df.columns:
                if col.startswith('H.'):
                    try:
                        h_values.append(float(row[col]))
                    except ValueError:
                        h_values.append(0.0)

            # Collect UD values
            ud_values = []
            for col in df.columns:
                if col.startswith('UD.'):
                    try:
                        ud_values.append(float(row[col]))
                    except ValueError:
                        ud_values.append(0.0)

            # Pad sequences to target length
            h_values = pad_sequence(h_values, TARGET_LENGTH)
            ud_values = pad_sequence(ud_values, TARGET_LENGTH)

            # Store as space-separated string of numbers
            combined_row['H'] = ' '.join(map(str, h_values))
            combined_row['UD'] = ' '.join(map(str, ud_values))

            test_df = pd.concat([test_df, pd.DataFrame([combined_row])], ignore_index=True)

        # Create illegal user entries for training (240 samples)
        other_users_data = df[df['subject'] != user]
        if len(other_users_data) >= 240:
            illegal_train = other_users_data.sample(n=240, random_state=42)
        else:
            illegal_train = other_users_data.sample(n=240, replace=True, random_state=42)

        # Create illegal user entries for train validation (60 samples)
        remaining_data = other_users_data.drop(illegal_train.index)
        if len(remaining_data) >= 60:
            illegal_train_val = remaining_data.sample(n=60, random_state=42)
        else:
            illegal_train_val = remaining_data.sample(n=60, replace=True, random_state=42)

        # Create illegal user entries for test (100 samples)
        remaining_data = remaining_data.drop(illegal_train_val.index)
        if len(remaining_data) >= 100:
            illegal_test = remaining_data.sample(n=100, random_state=42)
        else:
            illegal_test = remaining_data.sample(n=100, replace=True, random_state=42)

        # Process training illegal samples
        for index, row in illegal_train.iterrows():
            combined_row = {
                'Login': user,
                'IsLegalUser': 0
            }

            # Collect H values
            h_values = []
            for col in df.columns:
                if col.startswith('H.'):
                    try:
                        h_values.append(float(row[col]))
                    except ValueError:
                        h_values.append(0.0)

            # Collect UD values
            ud_values = []
            for col in df.columns:
                if col.startswith('UD.'):
                    try:
                        ud_values.append(float(row[col]))
                    except ValueError:
                        ud_values.append(0.0)

            # Pad sequences to target length
            h_values = pad_sequence(h_values, TARGET_LENGTH)
            ud_values = pad_sequence(ud_values, TARGET_LENGTH)

            # Store as space-separated string of numbers
            combined_row['H'] = ' '.join(map(str, h_values))
            combined_row['UD'] = ' '.join(map(str, ud_values))

            train_df = pd.concat([train_df, pd.DataFrame([combined_row])], ignore_index=True)

        # Process train validation illegal samples
        for index, row in illegal_train_val.iterrows():
            combined_row = {
                'Login': user,
                'IsLegalUser': 0
            }

            # Collect H values
            h_values = []
            for col in df.columns:
                if col.startswith('H.'):
                    try:
                        h_values.append(float(row[col]))
                    except ValueError:
                        h_values.append(0.0)

            # Collect UD values
            ud_values = []
            for col in df.columns:
                if col.startswith('UD.'):
                    try:
                        ud_values.append(float(row[col]))
                    except ValueError:
                        ud_values.append(0.0)

            # Pad sequences to target length
            h_values = pad_sequence(h_values, TARGET_LENGTH)
            ud_values = pad_sequence(ud_values, TARGET_LENGTH)

            # Store as space-separated string of numbers
            combined_row['H'] = ' '.join(map(str, h_values))
            combined_row['UD'] = ' '.join(map(str, ud_values))

            train_val_df = pd.concat([train_val_df, pd.DataFrame([combined_row])], ignore_index=True)

        # Process test illegal samples
        for index, row in illegal_test.iterrows():
            combined_row = {
                'Login': user,
                'IsLegalUser': 0
            }

            # Collect H values
            h_values = []
            for col in df.columns:
                if col.startswith('H.'):
                    try:
                        h_values.append(float(row[col]))
                    except ValueError:
                        h_values.append(0.0)

            # Collect UD values
            ud_values = []
            for col in df.columns:
                if col.startswith('UD.'):
                    try:
                        ud_values.append(float(row[col]))
                    except ValueError:
                        ud_values.append(0.0)

            # Pad sequences to target length
            h_values = pad_sequence(h_values, TARGET_LENGTH)
            ud_values = pad_sequence(ud_values, TARGET_LENGTH)

            # Store as space-separated string of numbers
            combined_row['H'] = ' '.join(map(str, h_values))
            combined_row['UD'] = ' '.join(map(str, ud_values))

            test_df = pd.concat([test_df, pd.DataFrame([combined_row])], ignore_index=True)

    # Shuffle all datasets
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_val_df = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV files
    train_df.to_csv('data/ML_KeystrokeData_train.csv', index=False)
    train_val_df.to_csv('data/ML_KeystrokeData_train_val.csv', index=False)
    test_df.to_csv('data/ML_KeystrokeData_test.csv', index=False)

    print("ML data preparation completed.")
    print("\nTraining Data:")
    print(f"Total rows: {len(train_df)}")
    print(f"Legal users (1): {len(train_df[train_df['IsLegalUser'] == 1])}")
    print(f"Illegal users (0): {len(train_df[train_df['IsLegalUser'] == 0])}")

    print("\nTraining Validation Data:")
    print(f"Total rows: {len(train_val_df)}")
    print(f"Legal users (1): {len(train_val_df[train_val_df['IsLegalUser'] == 1])}")
    print(f"Illegal users (0): {len(train_val_df[train_val_df['IsLegalUser'] == 0])}")

    print("\nTest Data:")
    print(f"Total rows: {len(test_df)}")
    print(f"Legal users (1): {len(test_df[test_df['IsLegalUser'] == 1])}")
    print(f"Illegal users (0): {len(test_df[test_df['IsLegalUser'] == 0])}")

    print(f"\nFeature length (H and UD): {TARGET_LENGTH}")

if __name__ == "__main__":
    prepare_ml_data()
