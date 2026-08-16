import torch
import numpy as np
from sklearn.svm import OneClassSVM

def OC_SVM(X_train, X_test, nu=0.01, gamma="scale"):

    # (N, Windows, Channels) -> (N, Features) -> (N,)

    X_train_flat = X_train.reshape(X_train.shape[0], -1) # (N_train, Windows, Channels) -> (N_train, Windows*Channels)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

    model.fit(X_train_flat)
    
    anomaly_scores = -model.score_samples(X_test_flat)

    return model, anomaly_scores