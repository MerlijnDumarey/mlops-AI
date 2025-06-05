import os
import argparse
from utils import (
    get_dataframe,
    remove_duplicates,
    flag_joint_missingness,
    impute_missingness,
    shuffle_data,
    split_data_by_label,
    save_data_to_dataset,
    dataframe_to_h5
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--train_label", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    df_train = get_dataframe(args)

    df_train_no_dup = remove_duplicates(df_train)

    df_train_missing_flagged = flag_joint_missingness(df_train_no_dup, df_train_no_dup.columns[:-1])

    df_train_imputed = impute_missingness(df_train_missing_flagged)
    
    df_train_shuffled = shuffle_data(df_train_imputed)

    label_dataframes = split_data_by_label(df_train_shuffled)

    for label, df_subset in label_dataframes.items():
        np_data = dataframe_to_h5(df_subset)
        save_data_to_dataset(args.output, np_data, f"label_{int(label)}")

if __name__ == "__main__":
    main()