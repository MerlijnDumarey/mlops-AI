import argparse
import os
from glob import glob
import h5py
import pandas as pd
import numpy as np
from utils import h5_to_dataframe, create_label_dummies, dataframe_to_h5
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def load_h5_data(folder_path):
    dataframes = []
    for file_name in sorted(os.listdir(folder_path)):
        print(f"file name: {file_name}")
        if not file_name.startswith("label_"):
            continue
        label = int((file_name.split("_")[1]).split(".")[0])
        h5_path = os.path.join(folder_path, file_name)
        with h5py.File(h5_path, "r") as f:
            data = f["data"][:]
        dataframes.append(h5_to_dataframe(data, label, is_missing_included=True))
    return pd.concat(dataframes, ignore_index=True, sort=False)

def training(train_data, train_label, val_data, val_label, epochs, model_path):
    print("TRAINING STARTED")
    early_stop = EarlyStopping(patience=15, monitor="val_loss", restore_best_weights=True)
    cb_save_best_model = ModelCheckpoint(filepath=model_path,
                                                            monitor='val_loss', 
                                                            save_best_only=True, 
                                                            verbose=1)

    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(2, 3), activation='relu',
                         kernel_initializer='glorot_uniform',
                         input_shape=(15, 4, 25, 1), return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())

    for i in range(2):
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_data, train_label,
        validation_data=(val_data, val_label),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop, cb_save_best_model],
        verbose=0
    )

    print(f"val_label shape: {val_label.shape}")

    val_preds = model.predict(val_data)
    predicted_classes = np.argmax(val_preds, axis=1)
    true_classes = np.argmax(val_label.to_numpy(), axis=1)

    print(f"\nClassification Report for model with 2 Dense layers:")
    report = classification_report(true_classes, predicted_classes, digits=4)
    print(report)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splitted-data-folder', type=str, dest='splitted_data_folder', help='Folder mounting point containing 3 subfolders: train, val and test')
    parser.add_argument('--output-folder', type=str, dest='output_folder', help='Output folder')
    parser.add_argument('--epochs', type=int, dest='epochs', help='The amount of epochs to train')
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    training_folder = os.path.join(args.splitted_data_folder, "train")
    print('Training folder:', training_folder)

    validation_folder = os.path.join(args.splitted_data_folder, "val")
    print('Validation folder:', validation_folder)

    testing_folder = os.path.join(args.splitted_data_folder, "test")
    print('Testing folder:', testing_folder)

    output_folder = args.output_folder
    print('Testing folder:', output_folder)
    
    MAX_EPOCHS = args.epochs
    
    model_name = "movement-convlstm"

    model_path = os.path.join(output_folder, model_name)
    os.makedirs(model_path, exist_ok=True)


    # As we're mounting the training_folder and testing_folder onto the `/mnt/data` directories, we can load in the images by using glob.
    # training_paths = glob(training_folder + "/*.h5", recursive=True)
    # validation_paths = glob(validation_folder + "/*.h5", recursive=True)
    # testing_paths = glob(testing_folder + "/*.h5", recursive=True)

    df_training = load_h5_data(training_folder)
    df_validation = load_h5_data(validation_folder)
    df_testing = load_h5_data(testing_folder)

    print("Training shape:", df_training.shape)
    print("Validation shape:", df_validation.shape)
    print("Testing shape:", df_testing.shape)

    print("Original training top 5")
    print(df_training.head(5))

    df_training_dummies = create_label_dummies(df_training)
    df_validation_dummies = create_label_dummies(df_validation)
    df_testing_dummies = create_label_dummies(df_testing)


    df_train_shuffled = df_training_dummies.sample(frac=1)
    df_val_shuffled = df_validation_dummies.sample(frac=1)
    df_test_shuffled = df_testing_dummies.sample(frac=1)
    
    print("Top five training records:")
    print(df_train_shuffled.head(5))
    print("Top five validataion records:")
    print(df_val_shuffled.head(5))
    print("Top five testing records:")
    print(df_test_shuffled.head(5))

    label_columns = [f'label_{i}' for i in range(10)]

    train_data_cleaned = dataframe_to_h5(df_train_shuffled)
    train_label_cleaned = df_train_shuffled[label_columns].copy().astype(np.uint8)
    val_data_cleaned = dataframe_to_h5(df_val_shuffled)
    val_label_cleaned = df_val_shuffled[label_columns].copy().astype(np.uint8)
    test_data_cleaned = dataframe_to_h5(df_test_shuffled)
    test_label_cleaned = df_test_shuffled[label_columns].copy().astype(np.uint8)

    print(f"train_data_cleaned shape: {train_data_cleaned.shape}")
    print(f"val_label_cleaned shape: {val_label_cleaned.shape}")
    print("val_label_cleaned top 5:")
    print(val_label_cleaned.head(5))

    train_data = train_data_cleaned.reshape((-1, 15, 4, 25, 1))
    val_data = val_data_cleaned.reshape((-1, 15, 4, 25, 1))
    test_data = test_data_cleaned.reshape((-1, 15, 4, 25, 1))


    model = training(train_data, train_label_cleaned, val_data, val_label_cleaned, MAX_EPOCHS, model_path)
    model.save(os.path.join(model_path, "final_model.h5"))

    print("DONE TRAINING")

    test_loss, test_accuracy = model.evaluate(test_data, test_label_cleaned, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    # Confusion matrix
    test_preds = model.predict(test_data)
    predicted_classes_test = np.argmax(test_preds, axis=1)
    true_classes_test = np.argmax(test_label_cleaned.to_numpy(), axis=1)

    cf_matrix = confusion_matrix(true_classes_test, predicted_classes_test)
    print(cf_matrix)
    
if __name__ == "__main__":
    main()