import torch
from torch import nn
import numpy as np
import pandas as pd
import os
import argparse
import joblib
from tqdm import tqdm
from dtaianomaly.evaluation import (
    AreaUnderROC, AreaUnderPR,
    VolumeUnderROC, VolumeUnderPR,
    RangeBasedPrecision, RangeBasedRecall, RangeBasedFBeta,
    AffiliationPrecision, AffiliationRecall, AffiliationFBeta
)
from sklearn.metrics import precision_recall_curve
from data.preprocess import preprocess
from data.dataset import SMDDataset
from data.dataset import get_dataloaders
from baselines.isolation_forest import Isolation_Forest
from baselines.oc_svm import OC_SVM
from baselines.lstm_ae import LSTMAutoencoder
from baselines.patchtst import PatchTST

def evaluate(anomaly_scores, labels, algo, window_size):

    labels = np.array(labels)
    anomaly_scores = np.array(anomaly_scores)

    roc_auc = AreaUnderROC().compute(labels, anomaly_scores)
    pr_auc = AreaUnderPR().compute(labels, anomaly_scores)
    
    vus_roc = VolumeUnderROC().compute(labels, anomaly_scores)
    vus_pr = VolumeUnderPR().compute(labels, anomaly_scores)

    precisions, recalls, thresholds = precision_recall_curve(labels, anomaly_scores)
    p_sliced = precisions[:-1]
    r_sliced = recalls[:-1]
    
    f1_scores = (2 * p_sliced * r_sliced) / (p_sliced + r_sliced + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
  
    best_predictions = (anomaly_scores >= best_threshold).astype(int)

    if np.sum(best_predictions) == 0:
        aff_precision, aff_recall, aff_f1 = 0.0, 0.0, 0.0
        rb_precision, rb_recall, rb_f1 = 0.0, 0.0, 0.0
    else:
        aff_precision = AffiliationPrecision().compute(labels, best_predictions)
        aff_recall = AffiliationRecall().compute(labels, best_predictions)
        aff_f1 = AffiliationFBeta().compute(labels, best_predictions)

        rb_precision = RangeBasedPrecision().compute(labels, best_predictions)
        rb_recall = RangeBasedRecall().compute(labels, best_predictions)
        rb_f1 = RangeBasedFBeta().compute(labels, best_predictions)

    results_dict = {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "VUS_ROC": vus_roc,
        "VUS_PR": vus_pr,
        "Affiliation_Precision": aff_precision,
        "Affiliation_Recall": aff_recall,
        "Affiliation_F1": aff_f1,
        "RangeBased_Precision": rb_precision,
        "RangeBased_Recall": rb_recall,
        "RangeBased_F1": rb_f1
    }

    for metric, value in results_dict.items():
        print(f"{metric.ljust(25)} : {value:.3f}")


def run_baselines_test(train_path, test_path, test_label_path, algo, window_size=100):

    print("Transformer-Based Anomaly Detection Using Time Series Foundation Models Baseline")

    train_data, test_data, test_label_data = preprocess(train_path, test_path, test_label_path)

    train_dataset = SMDDataset(train_data, window_size, is_channel_independent=False)
    test_dataset = SMDDataset(test_data, window_size, is_channel_independent=False)
    test_labels = test_label_data[window_size - 1 :]

    if algo in ["isolation_forest", "oc_svm", "all"] :

        X_train = torch.stack([train_dataset[i] for i in range(len(train_dataset))]).numpy()
        X_test = torch.stack([test_dataset[i] for i in range(len(test_dataset))]).numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (algo == "isolation_forest" or algo =="all"):

        print("Isolation Forest Baseline :")

        model, isolationforest_anomaly_scores = Isolation_Forest(X_train, X_test, n_estimators=100, contamination="auto", max_samples=256)

        print("Isolation Model Baseline Saved.")

        joblib.dump(model, "./models/baseline_models/isolation_forest.joblib")

        evaluate(isolationforest_anomaly_scores, test_labels, "Isolation Forest", window_size)

    if (algo == "oc_svm" or algo =="all"):

        print("OneClass SVM Baseline :")

        model, ocsvm_anomaly_scores = OC_SVM(X_train, X_test, nu=0.01, gamma="scale")

        print("OneClass SVM Baseline Saved.")
        
        joblib.dump(model, "./models/baseline_models/oc_svm.joblib")

        evaluate(ocsvm_anomaly_scores, test_labels, "OneClass SVM", window_size)

    if (algo == "lstm_ae" or algo =="all"):

        print("LSTM Autoencoder Baseline :")

        num_epochs = 10
        learning_rate = 1e-4
        batch_size = 128

        train_dataloader, test_dataloader = get_dataloaders(train_data, test_data, window_size, num_workers=0, pin_memory=False, batch_size=batch_size, is_channel_independent=False)

        hidden_dims = 64
        latent_dim = 16

        lstm_autoencoder = LSTMAutoencoder(
            n_channels=38,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_layers=2,
            dropout=0.1
        ).to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(params=lstm_autoencoder.parameters(), lr=learning_rate)

        for epoch in tqdm(range(num_epochs)):

            train_losses = []

            for idx, batch in enumerate(train_dataloader):

                batch = batch.to(device)

                reconstructed = lstm_autoencoder(batch)

                loss = loss_fn(reconstructed, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            average_training_loss = np.mean(train_losses)

            print(f"Epochs : {epoch+1}/{num_epochs} | Average Training Loss : {average_training_loss:.3f}")

        torch.save(lstm_autoencoder.state_dict(), "./models/baseline_models/lstm_autoencoder.pth")

        print("LSTM Autoencoder Baseline Saved.")

        lstmae_anomaly_scores = []

        lstm_autoencoder.eval()

        with torch.inference_mode():

            for idx, batch in enumerate(test_dataloader):

                batch = batch.to(device)

                batch_scores = lstm_autoencoder.anomaly_score(batch)
                lstmae_anomaly_scores.extend(batch_scores.cpu().numpy())

        lstmae_anomaly_scores = np.array(lstmae_anomaly_scores)

        evaluate(lstmae_anomaly_scores, test_labels, "LSTM Autoencoder", window_size)

    if (algo == "patchtst" or algo =="all"):

        print("PatchTST Baseline :")

        num_epochs = 10
        learning_rate = 1e-4
        batch_size = 128

        train_dataloader, test_dataloader = get_dataloaders(train_data, test_data, window_size,  num_workers=0, pin_memory=False, batch_size=batch_size, is_channel_independent=True)

        patch_length = 10
        stride = 10
        d_model = 128
        d_ff = 256
        n_heads = 4
        n_layers = 3

        patchtst = PatchTST(
            n_channels=38,
            T=window_size,
            patch_length=patch_length,
            stride=stride,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers
        ).to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(params=patchtst.parameters(), lr=learning_rate)

        for epoch in tqdm(range(num_epochs)):

            train_losses = []

            for idx, batch in enumerate(train_dataloader):

                batch = batch.to(device)

                reconstructed = patchtst(batch)

                loss = loss_fn(reconstructed, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            average_training_loss = np.mean(train_losses)

            print(f"Epochs : {epoch+1}/{num_epochs} | Average Training Loss : {average_training_loss:.3f}")

        torch.save(patchtst.state_dict(), "./models/baseline_models/patchtst.pth")

        print("PatchTST Baseline Saved.")

        patchtst_anomaly_scores = []

        patchtst.eval()

        with torch.inference_mode():

            for idx, batch in enumerate(test_dataloader):

                batch = batch.to(device)

                batch_scores = patchtst.anomaly_score(batch)
                patchtst_anomaly_scores.extend(batch_scores.cpu().numpy())

            patchtst_anomaly_scores = np.array(patchtst_anomaly_scores)

        evaluate(patchtst_anomaly_scores, test_labels, "PatchTST", window_size)

def main():

    parser = argparse.ArgumentParser(description="Baselines")

    parser.add_argument("--algo", type=str, choices=["isolation_forest", "oc_svm", "lstm_ae", "patchtst", "all"], default="all", help="Algorithm | Default runs all.")
    parser.add_argument("--window_size", type=int, default=100, help="Sliding Window Size | Default is 100.")

    args = parser.parse_args()

    algo = args.algo
    window_size = args.window_size

    train_path = "./ServerMachineDataset/train/machine-1-1.txt"
    test_path = "./ServerMachineDataset/test/machine-1-1.txt"
    test_label_path = "./ServerMachineDataset/test_label/machine-1-1.txt"

    run_baselines_test(train_path, test_path, test_label_path, algo, window_size)


if __name__=="__main__":

    main()