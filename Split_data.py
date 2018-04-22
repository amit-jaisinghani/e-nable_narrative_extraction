import pandas as pd
from sklearn.model_selection import train_test_split
import sys

data_set = pd.read_csv(sys.argv[1])

train, test = train_test_split(data_set, test_size=0.2)

# Saving the Data file
train.to_csv("./data/train.csv")
test.to_csv("./data/test.csv")
