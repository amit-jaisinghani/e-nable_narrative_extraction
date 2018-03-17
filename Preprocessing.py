"""
    Preprocesses the e-NABLE facebook posts using GloVe word Embeddings
    author: Amit Jaisinghani
"""
import pandas as pd
import numpy as np
import spacy


def loadGloVe():
    """
    Loads the GloVe embeddings to memory for easy access.
    Makes use of spacey's Glove binaries.
    Usage:
        python -m spacy download en_vectors_web_lg
    :return: Spacy object
    """
    print("Loading GloVe vectors to memory...")
    nlp = spacy.load('en_vectors_web_lg')
    print("GloVe Data Loaded!")
    return nlp


def loadData():
    """
    Takes a csv file and outputs a pandas dataframe.
    :param csvpath:
    :return:
    """
    csvpath = "./data/labelled_data/akshai_labels.csv"
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
    nlp = loadGloVe()
    df["y_term"] = df.apply(__rowToVec, axis=1).apply(np.array)

    df["x_term"] = df["content"].apply(lambda c: [token.vector for token in nlp(c)])
    df = df[df["x_term"].map(len) < 1000] # Remove vectors which have higher dimentions

    # Split dataframe as a random sample to 60:20:20 as train:validate:test set.
    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    return train, validate, test


def main():
    data_set = loadData()
    train, validate, test = getTrainTest(data_set)
    count = 5

    # for x in range(count):
    print(type(train["x_term"].as_matrix(columns=None)))

    pass


if __name__ == '__main__':
    main()
