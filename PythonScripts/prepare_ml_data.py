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

    if len(hold_times) < n: # Need at least n hold times for n-graphs
        return None

    # For n-graphs (n > 1), we need n hold times and n-1 between_times.
    # The number of between_times is typically one less than hold_times.
    # So, if len(hold_times) is L, len(between_times) should be L-1.
    # For an n-graph, we look at a sequence of n key presses.
    # This involves n hold times and n-1 between_times *within that n-key sequence*.

    # Example: For a 2-graph (bigraph), from H1, UD1, H2, UD2, H3...
    # First bigraph uses H1, UD1, H2.
    # Second bigraph uses H2, UD2, H3.
    # Number of n-graphs possible is len(hold_times) - n + 1.
    # We also need enough between_times. If len(hold_times) = L, we typically have L-1 between_times.
    # For the last n-graph, we need hold_times[L-n : L] and between_times[L-n : L-1].
    # So, we need len(between_times) to be at least (len(hold_times) - n + 1) + (n-1) - 1 = len(hold_times) - 1.
    # This is usually satisfied if between_times comes from the same full sequence.

    if len(between_times) < len(hold_times) -1 : # Basic check
        # print(f"Warning: Not enough between_times for hold_times for n={n}")
        # Decide how to handle, for now, let's return None if critical for n-graph calculations
        # For some n-graph features, it might be okay if we adjust loops, but UD definitely needs n-1
        pass # Let it proceed, DD and UU might still be formable for some parts. UD check is later.


    n_graph_between_keys_down = [] # DD
    n_graph_between_keys_up = []   # UU
    n_graph_hold = []              # H
    n_graph_between_keys = []      # UD (original definition)

    num_possible_n_graphs = len(hold_times) - n + 1

    for i in range(num_possible_n_graphs):
        # H (Sum of n consecutive hold times)
        n_graph_hold.append(sum(hold_times[i : i + n]))

        # DD (Down-Down): H_i + UD_i + H_{i+1} + ... + UD_{i+n-2} + H_{i+n-1}
        # This is sum of (H_j + UD_j) for j from i to i+n-2, then add H_{i+n-1}
        # Or, more simply, it's the time from first key down to last key down in the n-graph.
        # It involves n hold times and n-1 between_times.
        if i + (n - 1) < len(hold_times) and i + (n - 1) <= len(between_times): # Check UD bounds for DD
            current_dd = 0
            for j in range(n - 1): # Sum H_k and UD_k for k from i to i+n-2
                current_dd += hold_times[i+j]
                current_dd += between_times[i+j]
            current_dd += hold_times[i+n-1] # Add the last hold time
            n_graph_between_keys_down.append(current_dd)
        else:
            # Not enough data for full DD, decide on padding or skipping. For now, let's try to fill if possible.
            # This case should ideally be prevented by the initial length checks or handled by padding if desired.
            # If an n-graph cannot be formed, it should have been caught by "if len(hold_times) < n:"
            # Or if we expect partial n-graphs, this logic would be different.
            # Given the problem context, we expect full n-graphs.
            # For safety, if we somehow reach here, append a default (e.g. 0 or handle as error)
             n_graph_between_keys_down.append(0.0)


        # UU (Up-Up): UD_i + H_{i+1} + UD_{i+1} + ... + H_{i+n-1} + UD_{i+n-1}
        # Time from first key up to last key up.
        # Involves n-1 between_times and n-1 intermediate hold times, plus the first between_time.
        # More accurately: UD_i + (H_{i+1} + UD_{i+1}) + ... + (H_{i+n-1} + UD_{i+n-1})
        # This needs (n-1) UDs starting from UD_i, and (n-1) Hs starting from H_{i+1}
        if i + (n - 1) <= len(between_times) and i + n < len(hold_times) +1 : # Check bounds for UU
            current_uu = 0
            for j in range(n - 1): # Sum UD_k and H_{k+1} for k from i to i+n-2
                current_uu += between_times[i+j]
                current_uu += hold_times[i+j+1]
            n_graph_between_keys_up.append(current_uu)
        else:
             n_graph_between_keys_up.append(0.0)


        # UD (Sum of n-1 consecutive between_times)
        if n > 1:
            if i + (n - 1) <= len(between_times): # Check if enough UDs exist starting from index i
                n_graph_between_keys.append(sum(between_times[i : i + n - 1]))
            else:
                # Not enough UD times for this n-graph starting at i
                # This implies an issue or the need for padding if partials are allowed.
                # For now, append 0.0, assuming we need a value for each potential n-graph start.
                n_graph_between_keys.append(0.0)
        # For n=1, UD is handled by the initial block. Here we only care for n > 1.

    if n == 1: # Should have been caught by the first if, but as a fallback
        return {'H': hold_times, 'UD': between_times}

    # Ensure all feature lists have the same length, corresponding to num_possible_n_graphs
    # This might require padding if some calculations failed due to boundary conditions
    # For now, the try-append(0.0) handles cases where a specific feature can't be calculated for an n-graph

    # Final check for list lengths, if they are not num_possible_n_graphs, it's an issue.
    # Example: if len(n_graph_hold) is 10, all others should be 10.
    # This should be inherently true if append(0.0) is used for missing ones.

    return {
        'H': n_graph_hold,
        'DD': n_graph_between_keys_down,
        'UU': n_graph_between_keys_up,
        'UD': n_graph_between_keys if n > 1 else [] # UD is empty for 1-graph here as it's separate
    }

def prepare_ml_data():
    df = pd.read_csv('data/DSL-StrongPasswordData.csv')
    TARGET_LENGTH = 20

    train_dfs = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
    train_val_dfs = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
    test_dfs = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}

    unique_users = df['subject'].unique()

    # Define how many samples to take from each OTHER user for illegal attempts
    SAMPLES_PER_OTHER_USER_MAP = {
        'train': 5,  # Aim for ~240 illegal samples if 50 other users (50*5=250)
        'train_val': 1, # Aim for ~60 illegal samples (50*1=50, adjust if needed)
        'test': 2    # Aim for ~100 illegal samples (50*2=100)
    }
    # Small adjustment if unique_users is small, to avoid division by zero or tiny numbers
    num_other_users = len(unique_users) - 1 if len(unique_users) > 1 else 1


    def pad_sequence(values, target_length):
        if len(values) < target_length:
            return values + [0.0] * (target_length - len(values))
        elif len(values) > target_length:
            return values[:target_length]
        return values

    for user in unique_users:
        user_data = df[df['subject'] == user]

        # Ensure enough data for sampling, otherwise, adjust counts or skip user for splitting
        if len(user_data) < (240 + 60 + 10): # Min for train, val, and some test
            print(f"Warning: User {user} has insufficient data ({len(user_data)} rows) for standard split. Skipping or adjusting.")
            # Decide: skip user, or take all for one category, or proportionally reduce.
            # For now, let's try to be robust if possible, but it might lead to empty sets.
            # A simple approach: if too few, put all in train, and they won't have val/test.
            # This needs careful handling for illegal sample generation too.
            # For now, the original sampling logic for user's own data might fail if n is too large.
            # Let's assume enough data based on original script's premise.

        user_train = user_data.sample(n=min(240, len(user_data)), random_state=42)
        remaining_data = user_data.drop(user_train.index)
        user_train_val = remaining_data.sample(n=min(60, len(remaining_data)), random_state=42)
        remaining_data_for_test = remaining_data.drop(user_train_val.index)
        user_test = remaining_data_for_test.sample(n=min(100, len(remaining_data_for_test)), random_state=42)


        for dataset_name, data_split in [('train', user_train), ('train_val', user_train_val), ('test', user_test)]:
            if data_split.empty: continue

            # Process legal entries
            for index, row in data_split.iterrows():
                h_values = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in df.columns if col.startswith('H.')]
                ud_values = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in df.columns if col.startswith('UD.')]

                for n_graph_level in [1, 2, 3]:
                    n_graph_features = calculate_ngraph_features(h_values, ud_values, n_graph_level)
                    if n_graph_features is None: continue

                    combined_row = {'Login': user, 'IsLegalUser': 1}
                    for feature_type, values in n_graph_features.items():
                        if n_graph_level == 1 and feature_type not in ['H', 'UD']: continue
                        if values is None: values = [] # Should not happen if n_graph_features is not None
                        padded_values = pad_sequence(values, TARGET_LENGTH)
                        combined_row[feature_type] = ' '.join(map(str, padded_values))

                    if not combined_row.get('H') and not combined_row.get('UD'): # Skip if no features produced
                        continue

                    if dataset_name == 'train':
                        train_dfs[n_graph_level] = pd.concat([train_dfs[n_graph_level], pd.DataFrame([combined_row])], ignore_index=True)
                    elif dataset_name == 'train_val':
                        train_val_dfs[n_graph_level] = pd.concat([train_val_dfs[n_graph_level], pd.DataFrame([combined_row])], ignore_index=True)
                    else:
                        test_dfs[n_graph_level] = pd.concat([test_dfs[n_graph_level], pd.DataFrame([combined_row])], ignore_index=True)

            # Create illegal user entries for the current 'user' (target) and 'dataset_name'
            other_individual_users = [u for u in unique_users if u != user]
            if not other_individual_users: continue

            samples_to_take_per_other = SAMPLES_PER_OTHER_USER_MAP[dataset_name]

            for other_impostor_user in other_individual_users:
                data_from_one_other_user = df[df['subject'] == other_impostor_user]
                if data_from_one_other_user.empty: continue

                num_actual_samples_from_this_other = min(samples_to_take_per_other, len(data_from_one_other_user))
                if num_actual_samples_from_this_other == 0: continue

                # Sample WITHOUT replacement from this specific other user's data
                illegal_samples_from_this_other = data_from_one_other_user.sample(n=num_actual_samples_from_this_other, random_state=42, replace=False)

                for index, row in illegal_samples_from_this_other.iterrows():
                    h_values = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in df.columns if col.startswith('H.')]
                    ud_values = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in df.columns if col.startswith('UD.')]

                    for n_graph_level in [1, 2, 3]:
                        n_graph_features = calculate_ngraph_features(h_values, ud_values, n_graph_level)
                        if n_graph_features is None: continue

                        combined_row = {'Login': user, 'IsLegalUser': 0} # Target user, but illegal
                        for feature_type, values in n_graph_features.items():
                            if n_graph_level == 1 and feature_type not in ['H', 'UD']: continue
                            if values is None: values = []
                            padded_values = pad_sequence(values, TARGET_LENGTH)
                            combined_row[feature_type] = ' '.join(map(str, padded_values))

                        if not combined_row.get('H') and not combined_row.get('UD'): continue

                        if dataset_name == 'train':
                            train_dfs[n_graph_level] = pd.concat([train_dfs[n_graph_level], pd.DataFrame([combined_row])], ignore_index=True)
                        elif dataset_name == 'train_val':
                            train_val_dfs[n_graph_level] = pd.concat([train_val_dfs[n_graph_level], pd.DataFrame([combined_row])], ignore_index=True)
                        else: # test
                            test_dfs[n_graph_level] = pd.concat([test_dfs[n_graph_level], pd.DataFrame([combined_row])], ignore_index=True)

    # Shuffle and save datasets
    for n_graph_level in [1, 2, 3]:
        if not train_dfs[n_graph_level].empty:
            train_dfs[n_graph_level] = train_dfs[n_graph_level].sample(frac=1, random_state=42).reset_index(drop=True)
        if not train_val_dfs[n_graph_level].empty:
            train_val_dfs[n_graph_level] = train_val_dfs[n_graph_level].sample(frac=1, random_state=42).reset_index(drop=True)
        if not test_dfs[n_graph_level].empty:
            test_dfs[n_graph_level] = test_dfs[n_graph_level].sample(frac=1, random_state=42).reset_index(drop=True)

        train_dfs[n_graph_level].to_csv(f'data/ML_KeystrokeData_train_{n_graph_level}graph.csv', index=False)
        train_val_dfs[n_graph_level].to_csv(f'data/ML_KeystrokeData_train_val_{n_graph_level}graph.csv', index=False)
        test_dfs[n_graph_level].to_csv(f'data/ML_KeystrokeData_test_{n_graph_level}graph.csv', index=False)

        print(f"\n{n_graph_level}-Graph Data (New Method):")
        print(f"Training Data: Total rows: {len(train_dfs[n_graph_level])}, Legal: {len(train_dfs[n_graph_level][train_dfs[n_graph_level]['IsLegalUser'] == 1]) if not train_dfs[n_graph_level].empty else 0}, Illegal: {len(train_dfs[n_graph_level][train_dfs[n_graph_level]['IsLegalUser'] == 0]) if not train_dfs[n_graph_level].empty else 0}")
        print(f"Training Validation Data: Total rows: {len(train_val_dfs[n_graph_level])}, Legal: {len(train_val_dfs[n_graph_level][train_val_dfs[n_graph_level]['IsLegalUser'] == 1]) if not train_val_dfs[n_graph_level].empty else 0}, Illegal: {len(train_val_dfs[n_graph_level][train_val_dfs[n_graph_level]['IsLegalUser'] == 0]) if not train_val_dfs[n_graph_level].empty else 0}")
        print(f"Test Data: Total rows: {len(test_dfs[n_graph_level])}, Legal: {len(test_dfs[n_graph_level][test_dfs[n_graph_level]['IsLegalUser'] == 1]) if not test_dfs[n_graph_level].empty else 0}, Illegal: {len(test_dfs[n_graph_level][test_dfs[n_graph_level]['IsLegalUser'] == 0]) if not test_dfs[n_graph_level].empty else 0}")
        print(f"Feature length: {TARGET_LENGTH}")

if __name__ == "__main__":
    prepare_ml_data()
