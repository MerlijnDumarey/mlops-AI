import os
import h5py
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

def h5_to_dataframe(data, labels=None, is_missing_included=False):
    n_samples, n_frames, n_features = data.shape
    flat_data = data.reshape(n_samples, -1)  # flatten to (N, 1125)
    
    # Generate column names: frame_0_joint_0_x, ..., frame_14_joint_24_z
    columns = []
    axes = ['x', 'y', 'z']
    if is_missing_included:
        axes.append('is_missing')
    for t in range(n_frames):
        for j in range(25):
            for axis in axes:
                columns.append(f"frame_{t}_joint_{j}_{axis}")

    df = pd.DataFrame(flat_data, columns=columns)
    if labels is not None:
        df['label'] = labels
    return df

def get_dataframe(args):
    with h5py.File(args.train_data, "r") as f:
        X_train = f["data"][:]

    with h5py.File(args.train_label, "r") as f:
        y_train = f["label"][:]

    return h5_to_dataframe(X_train, y_train)

def dataframe_to_h5(df):
    """
    Convert a DataFrame to an H5-compatible NumPy array with the correct dimensions.
    Args:
        df: Input DataFrame.
        autoencoded_features: Number of autoencoded features to include.
    Returns:
        Reshaped NumPy array.
    """
    # Drop label columns if they exist
    if 'label_0' in df.columns:
        df = df.drop(columns=[f'label_{i}' for i in range(10)])
    
    n_samples = df.shape[0]
    n_frames = 15
    n_joints = 25
    n_features_per_joint = 4  # 3 coordinates + 1 missingness flag
    n_features_per_frame = n_joints * n_features_per_joint
    total_features = n_features_per_frame

    # Convert DataFrame to NumPy array and reshape
    flat_data = df.values.astype(np.float32)  # shape (n_samples, total_features * n_frames)
    data = flat_data.reshape(n_samples, n_frames, total_features)
    return data

def flag_joint_missingness(df, target_cols):
    df = df.copy()
    joint_cols = {}

    # Step 1: Group target x/y/z columns by joint base name
    for col in target_cols:
        if col.endswith(('_x', '_y', '_z')):
            joint_base = '_'.join(col.split('_')[:-1])
            joint_cols.setdefault(joint_base, []).append(col)

    # Step 2: Create missing flag columns efficiently
    missing_flags = {
        f'{joint}_is_missing': df[cols].isna().any(axis=1)
        for joint, cols in joint_cols.items()
    }
    missing_flags_df = pd.DataFrame(missing_flags, index=df.index)

    # Step 3: Concatenate flags once to avoid fragmentation
    df = pd.concat([df, missing_flags_df], axis=1)

    # Step 4: Reorder target columns to: x, y, z, is_missing per joint
    ordered_target_cols = []
    for joint in sorted(joint_cols.keys(), key=lambda s: [int(part) for part in s.replace("frame_", "").replace("joint_", "").split('_')]):
        x_col = f'{joint}_x'
        y_col = f'{joint}_y'
        z_col = f'{joint}_z'
        flag_col = f'{joint}_is_missing'
        ordered_target_cols.extend([x_col, y_col, z_col, flag_col])

    # Step 5: Keep columns not in target list
    all_new_cols = set(ordered_target_cols)
    remaining_cols = [col for col in df.columns if col not in all_new_cols]

    # Step 6: Return final DataFrame with ordered target block first, rest after
    return df[ordered_target_cols + remaining_cols]

def remove_duplicates(df_train):
    df_train_no_dup = df_train.drop_duplicates(keep='first', subset=df_train.columns)

    df_train_no_dup = df_train_no_dup[~(df_train_no_dup.iloc[:, :-1] == 0).all(axis=1)]

    df_train_no_dup.iloc[:, :-1] = df_train_no_dup.iloc[:, :-1].replace(0, np.nan)

    return df_train_no_dup

def impute_missingness(df_train_missing_flagged):
    n_frames = 15
    n_joints = 25
    features = ["x", "y", "z", "is_missing"]

    # Construct the column names for x, y, and z
    x_cols = [f"frame_{f}_joint_{j}_x" for f in range(n_frames) for j in range(n_joints)]
    y_cols = [f"frame_{f}_joint_{j}_y" for f in range(n_frames) for j in range(n_joints)]
    z_cols = [f"frame_{f}_joint_{j}_z" for f in range(n_frames) for j in range(n_joints)]

    # Combine the lists of column names
    xyz_cols = x_cols + y_cols + z_cols

    # Select the x, y, z columns for imputation
    df_train_xyz = df_train_missing_flagged[xyz_cols].copy()

    scaler = StandardScaler()

    scaled_train_xyz = scaler.fit_transform(df_train_xyz)

    # mice_imputer = IterativeImputer(max_iter=10)

    mean_imputer = SimpleImputer(strategy="mean") 

    # Impute the scaled data
    # imputed_scaled_train_xyz = mice_imputer.fit_transform(scaled_train_xyz)
    imputed_scaled_train_xyz = mean_imputer.fit_transform(scaled_train_xyz)

    # inverse transform the imputed data back to the original scale
    imputed_train_xyz = scaler.inverse_transform(imputed_scaled_train_xyz)

    df_train_imputed = df_train_missing_flagged.copy()

    # and now we can replace the original columns with the imputed ones
    df_train_imputed[xyz_cols] = imputed_train_xyz

    return df_train_imputed

def create_label_dummies(df_train_imputed):
    dummies = pd.get_dummies(df_train_imputed.label.astype(np.int32), prefix='label')
    
    for i in range(10):
        col = f'label_{i}'
        if col not in dummies:
            dummies[col] = 0
            
    df_train_imputed = pd.concat([df_train_imputed, dummies], axis=1)
    df_train_imputed.drop(columns='label', inplace=True)

    num_feat = df_train_imputed.select_dtypes(np.number).columns

    scaler = StandardScaler()
    df_train_imputed[num_feat] = scaler.fit_transform(df_train_imputed[num_feat])

    return df_train_imputed

def shuffle_data(df_train_imputed):
    return df_train_imputed.sample(frac=1)

def split_data_by_label(df):
    label_groups = {}
    for label_value in sorted(df['label'].unique()):
        label_df = df[df['label'] == label_value].drop(columns='label').copy()
        label_groups[label_value] = label_df
    return label_groups

def save_data_to_dataset(base_output_dir, data, folder_name):
    folder_path = os.path.join(base_output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "data.h5")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
