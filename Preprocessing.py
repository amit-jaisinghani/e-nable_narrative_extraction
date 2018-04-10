"""
    Preprocesses the e-NABLE facebook posts using GloVe word Embeddings
    author: Amit Jaisinghani
"""
import pandas as pd
import numpy as np
import spacy


def loadData():
    """
    Takes a csv file and outputs a pandas dataframe.
    :param csvpath:
    :return:
    """
    csvpath = "./data/train.csv"
    columns = ["content", "Report", "Device", "Delivery", "Progress",
             "becoming_member", "attempt_action" , "Activity", "Other"]

    eNABLE_df = pd.read_csv(csvpath, usecols=columns, keep_default_na=False)
    return eNABLE_df


def __rowToVec(row):
    outputVector = (row["Report"], row["Device"], row["Delivery"], row["Progress"], row["becoming_member"],
                    row["attempt_action"], row["Activity"], row["Other"])
    return outputVector


def getTrainTest(df):
    """
    Generates files train and test csv
    :param testRate: Percentage of dataframe to be set aside for test data
    :return:
    """
    df["labels"] = df.apply(__rowToVec, axis=1).apply(np.array)
    # df["y_term"] = df["Report"]

    df = df[df["content"].map(len) < 1000]  # Remove vectors which have higher dimentions

    # Split dataframe as a random sample to 60:20:20 as train:validate:test set.
    # train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    # train = df.sample(frac=0.8, random_state=200)
    # test = df.drop(train.index)

    return df


def main():
    # data_set = loadData()
    # train, test = getTrainTest(data_set)
    #
    # print(train["x_term"])
    # print(train["y_term"])

    output = np.expand_dims(np.vstack([0, 0, 1, 1]), 1)
    print(output)
    pass


if __name__ == '__main__':
    main()
