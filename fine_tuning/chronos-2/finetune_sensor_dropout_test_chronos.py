import torch
from torch import nn
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from dtaianomaly.evaluation import (
    AreaUnderROC, AreaUnderPR,
    VolumeUnderROC, VolumeUnderPR,
    RangeBasedPrecision, RangeBasedRecall, RangeBasedFBeta,
    AffiliationPrecision, AffiliationRecall, AffiliationFBeta
)
from data.preprocess import preprocess
from data.dataset import SMDDataset
from data.dataset import get_dataloaders
from chronos import Chronos2Pipeline
from peft import PeftModel

class ChronosCrossMachineEvaluator(nn.Module):

    def __init__(self, saved_path="./models/chronos-2"):

        super().__init__()

        # Pretrained 

        self.pretrained_model = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="cuda",
            dtype=torch.float32
        )

        # Load Fine-Tuned Model

        self.model = PeftModel.from_pretrained(self.pretrained_model.model, saved_path)

        self.forecasting_horizon = 16

        self.quantiles = self.pretrained_model.model.quantiles

        self.model.eval()

    def crps(self, target_ci, quantile_preds):

        q = self.quantiles.to(target_ci.device)

        y = target_ci.unsqueeze(1) # (B*C, 1, 16)
        q = q.view(1, -1, 1) # (1, 21, 1)
        yhat = quantile_preds # (B*c, 21, 16)

        diff = y - yhat

        pinball = torch.where(
            diff >= 0,
            q*diff,
            (q-1)*diff
        )

        crps = pinball.mean(dim=(1,2)) # (B*C,)

        return crps

    def anomaly_score(self, x):

        B, T, C = x.shape

        x_ci = x.permute(0,2,1).reshape(B*C, T) # (B*C, T)
        context_ci = x_ci[:, :-self.forecasting_horizon] # (B*C, T-16)
        target_ci = x_ci[:, -self.forecasting_horizon:] # (B*C, 16)
    
        quantile_preds = self.forward(context_ci) # (B*C, 21, 16)
        
        crps = self.crps(target_ci, quantile_preds)

        crps = crps.reshape(B, C) # (B, C)

        anomaly_score = crps.mean(dim=1) # (B,)
        
        return anomaly_score

    def forward(self, x):

        outputs = self.model(x)

        return outputs.quantile_preds

def evaluate(anomaly_scores, labels, algo, sensor_dropout_rate):

    labels = np.array(labels)
    anomaly_scores = np.array(anomaly_scores)

    anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e9, neginf=-1e9)

    roc_auc = AreaUnderROC().compute(labels, anomaly_scores)
    pr_auc = AreaUnderPR().compute(labels, anomaly_scores)
    vus_roc = VolumeUnderROC().compute(labels, anomaly_scores)
    vus_pr = VolumeUnderPR().compute(labels, anomaly_scores)

    precisions, recalls, thresholds = precision_recall_curve(labels, anomaly_scores)
    p_sliced = precisions[:-1]
    r_sliced = recalls[:-1]
    
    f1_scores = (2 * p_sliced * r_sliced) / (p_sliced + r_sliced + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_predictions = (anomaly_scores > best_threshold).astype(int)

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

    print("\n" + "=" * 45)
    print(f" {algo.upper()} EVALUATION SCORECARD FOR SENSOR DROPOUT RATE : {sensor_dropout_rate}")
    print("=" * 45)
    for metric, value in results_dict.items():
        print(f"{metric.ljust(25)} : {value:.4f}")
    print("=" * 45)

def apply_sensor_dropout(test_data, dropout_rate=0.0):

    # Simulates physical sensor failure by zeroing out a percentage of the 38 SMD channels.

    if dropout_rate == 0.0:
        return test_data
        
    data_dropped = test_data.copy()
    num_sensors = data_dropped.shape[1] # 38 for SMD
    
    # Calculate how many sensors to mathematically 'kill'
    num_to_drop = int(num_sensors * dropout_rate)
    
    if num_to_drop > 0:
        # Randomly select which sensors fail
        sensors_to_drop = np.random.choice(num_sensors, num_to_drop, replace=False)
        print(f"[*] SENSOR DROPOUT ACTIVE: Dropping {num_to_drop} sensors: {sensors_to_drop}")
        
        # Overwrite the selected sensor columns with 0.0 (the normalized mean)
        data_dropped[:, sensors_to_drop] = 0.0
        
    return data_dropped

def main():

    print("Transformer-Based Anomaly Detection Using Time Series Foundation Models")

    parser = argparse.ArgumentParser(description="Evaluation")

    parser.add_argument("--sensor_dropout_rate", type=float, default=0.0, help="Sensor Dropout Rate | Default is 0.0.")

    args = parser.parse_args()

    sensor_dropout_rate = args.sensor_dropout_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using Device : {device}")

    window_size = 96
    batch_size = 128

    train_path = "./ServerMachineDataset/train/machine-1-3.txt"
    test_path = "./ServerMachineDataset/test/machine-1-3.txt"
    test_label_path = "./ServerMachineDataset/test_label/machine-1-3.txt"

    print("Preprocessing unseen server data...")
    train_data, test_data, test_label_data = preprocess(train_path, test_path, test_label_path)

    test_data = apply_sensor_dropout(test_data, sensor_dropout_rate)

    test_labels = test_label_data[window_size - 1 :]

    _, test_dataloader = get_dataloaders(
        train_data, test_data, window_size, 
        num_workers=0, pin_memory=False, batch_size=batch_size, is_channel_independent=False
    )

    # Load Model

    chronos_evaluator = ChronosCrossMachineEvaluator().to(device)

    print("Running Inference on Server 1-3 :")
    anomaly_scores = []
    
    with torch.inference_mode():
        for batch in tqdm(test_dataloader, desc="Evaluating Windows"):
            batch = batch.to(device)
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                batch_scores = chronos_evaluator.anomaly_score(batch)
            anomaly_scores.extend(batch_scores.cpu().numpy())

    evaluate(anomaly_scores, test_labels, "Chronos-2 (Train: 1-1 | Test: 1-3)", sensor_dropout_rate)

if __name__ == "__main__":
    main()