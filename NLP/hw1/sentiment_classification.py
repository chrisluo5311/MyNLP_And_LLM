import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer


def import_csv(file_path):
    file_size = os.path.getsize(file_path)
    print("file_size:", file_size)
    # raw_train_data = np.full(size, -1)
    # with open(file_path, 'r') as f:
    #     for each_line in f:
    #         text = each_line.split(",")[1].rstrip()
    #         print(text)
    #         size += 1



if __name__ == '__main__':
    x_train_path = "./train_data/x_train.csv"
    y_train_path = "./train_data/y_train.csv"
    x_test_path = "./test_data/x_test.csv"
    import_csv(x_train_path)