import pandas as pd
import numpy as np


def load_data(file):
    columns = ["content", "Report", "Device", "Delivery", "Progress",
             "becoming_member", "attempt_action", "Activity", "Other"]

    return pd.read_csv(file, usecols=columns, keep_default_na=False)


def row_to_vec(row):
    outputVector = ["\t"]
    if row["Report"] == 1:
        outputVector.append("Report")
    if row["Device"] == 1:
        outputVector.append("Device")
    if row["Delivery"] == 1:
        outputVector.append("Delivery")
    if row["Progress"] == 1:
        outputVector.append("Progress")
    if row["becoming_member"] == 1:
        outputVector.append("becoming_member")
    if row["attempt_action"] == 1:
        outputVector.append("attempt_action")
    if row["Activity"] == 1:
        outputVector.append("Activity")
    if row["Other"] == 1:
        outputVector.append("Other")
    outputVector.append("\n")
    return outputVector


def apply_filters(df, max_length):
    df["labels"] = df.apply(row_to_vec, axis=1).apply(np.array)
    df = df[df["content"].map(len) < max_length]
    return df


def get_train_validate_test_dataset(max_length):
    training_dataset = apply_filters(load_data("./data/train.csv"), max_length)
    validate_dataset = apply_filters(load_data("./data/validate.csv"), max_length)
    test_dataset = apply_filters(load_data("./data/test.csv"), max_length)
    return training_dataset, validate_dataset, test_dataset


def main():
    training_dataset, validate_dataset, test_dataset = get_train_validate_test_dataset(1000)
    print(training_dataset["labels"])
    pass


if __name__ == '__main__':
    main()
