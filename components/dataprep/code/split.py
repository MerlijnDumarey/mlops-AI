import os
import h5py
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

def split_and_save(data, label, output_dir, train_ratio, val_ratio):
    X_train, X_temp = train_test_split(data, test_size=(1 - train_ratio))
    X_val, X_test = train_test_split(X_temp, test_size=(1 - val_ratio/(1 - train_ratio)))

    sets = {'train': X_train, 'val': X_val, 'test': X_test}

    for split_name, split_data in sets.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        with h5py.File(os.path.join(split_dir, f"label_{label}.h5"), "w") as f:
            f.create_dataset("data", data=split_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--val_ratio", type=float)
    args = parser.parse_args()

    print(f"📂 Listing contents of input_dir: {args.input_dir}")
    print(os.listdir(args.input_dir))

    for folder_name in sorted(os.listdir(args.input_dir)):
        if not folder_name.startswith("label_"):
            continue
        label = int(folder_name.split("_")[1])
        h5_path = os.path.join(args.input_dir, folder_name, "data.h5")
        with h5py.File(h5_path, "r") as f:
            data = f["data"][:]
        split_and_save(data, label, args.output_dir, args.train_ratio, args.val_ratio)

    print("✅ Completed split. Output directory structure:")
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            print(os.path.join(root, file))

if __name__ == "__main__":
    main()
