import torch
import numpy as np
from sklearn.ensemble import IsolationForest

def Isolation_Forest(X_train, X_test, n_estimators=100, contamination="auto", max_samples=256):

    # (N, Windows, Channels) -> (N, Features) -> (N,)

    X_train_flat = X_train.reshape(X_train.shape[0], -1) # (N_train, Windows, Channels) -> (N_train, Windows*Channels)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, max_samples=max_samples)

    model.fit(X_train_flat)

    anomaly_scores = -model.score_samples(X_test_flat)

    return model, anomaly_scores