import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self):
        """初始化Logistic Regression模型"""
        self._theta = None

    # def transform(self, data):

    def sigmoid(self,r):
        return 1.0 / (1.0 + np.exp(-r))

    # def fit(self, X_train, y_train, eta=0.01, epsilon=1e-6, iters=1000):



if __name__ == '__main__':
    train_file_path = 'twitter_training.csv'
    val_file_path = 'twitter_validation.csv'

    df = pd.read_csv(train_file_path)
    print(df.head())

