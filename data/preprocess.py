import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess(train_path, test_path, test_label_path):

    train_data = np.loadtxt(train_path, delimiter=',')
    test_data = np.loadtxt(test_path, delimiter=',')
    test_label_data = np.loadtxt(test_label_path, delimiter=',')

    scaler = StandardScaler()

    train_data = scaler.fit_transform(train_data)

    test_data = scaler.transform(test_data)

    return train_data, test_data, test_label_data
