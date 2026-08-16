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
from peft import LoraConfig, get_peft_model

# Chronos-2 Fine-Tuned Anomaly Detector

class ChronosFineTunedAnomalyDetector(nn.Module):

    def __init__(self, n_channels):

        super().__init__()

        # Pretrained Time Series Foundation Model

        self.pretrained_model = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="cuda",
            dtype=torch.float32
        )

        # LoRA Configuration Parameters
    
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none"
        )

        # Model

        self.model = get_peft_model(self.pretrained_model.model, self.lora_config)

        self.forecasting_horizon = 16

        self.quantiles = self.pretrained_model.model.quantiles

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

def evaluate(anomaly_scores, labels, algo):

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
    print(f" {algo.upper()} EVALUATION SCORECARD")
    print("=" * 45)
    for metric, value in results_dict.items():
        print(f"{metric.ljust(25)} : {value:.4f}")
    print("=" * 45)

def main():

    print("Transformer-Based Anomaly Detection Using Time Series Foundation Models")

    parser = argparse.ArgumentParser(description="Fine-Tuning Time Series Foundation Models")
    parser.add_argument("--window_size", type=int, default=96, help="Sliding Window Size | Default is 96.")
    parser.add_argument("--few_shot_rate", type=float, default=0.0, help="Few Shot Rate | Default is 0.0 .")
    args = parser.parse_args()

    window_size = args.window_size
    few_shot_rate = args.few_shot_rate

    print(f"Few Shot Rate : {few_shot_rate:.1%}")

    num_epochs = 7
    learning_rate = 5e-5
    batch_size = 16

    train_path = "./ServerMachineDataset/train/machine-1-1.txt"
    test_path = "./ServerMachineDataset/test/machine-1-1.txt"
    test_label_path = "./ServerMachineDataset/test_label/machine-1-1.txt"

    train_data, test_data, test_label_data = preprocess(train_path, test_path, test_label_path)

    if 0.0 < few_shot_rate < 1.0 :

        num_samples = int(few_shot_rate * len(train_data))

        train_data = train_data[:num_samples]

    test_labels = test_label_data[window_size - 1 :]

    train_dataloader, test_dataloader = get_dataloaders(
        train_data, test_data, window_size, 
        num_workers=2, pin_memory=True, batch_size=batch_size, is_channel_independent=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device : {device}")

    chronos = ChronosFineTunedAnomalyDetector(n_channels=38).to(device)

    def pinball_loss(quantile_preds, target, quantiles):

        y = target.unsqueeze(1)              # (B*C, 1, 16)
        q = quantiles.view(1, -1, 1)         # (1, 21, 1)
        diff = y - quantile_preds            # (B*C, 21, 16)
        loss = torch.where(diff >= 0, q * diff, (q - 1) * diff)
        return loss.mean()

    optimizer = torch.optim.AdamW(params=chronos.parameters(), lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    steps_per_epoch = len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=num_epochs * steps_per_epoch,
        pct_start=0.3
    )

    print("Training :")
    for epoch in tqdm(range(num_epochs)):

        train_losses = []
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", unit="batch")

        for idx, batch in enumerate(train_dataloader):

            batch = batch.to(device)

            B, T, C = batch.shape

            batch_ci = batch.permute(0, 2, 1).reshape(B*C, T)

            context_ci = batch_ci[:, :-chronos.forecasting_horizon]
            target_ci = batch_ci[:, -chronos.forecasting_horizon:]

            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):

                quantile_preds = chronos(context_ci)
                forecasted_median_ci = quantile_preds[:, 10, :]
                loss = pinball_loss(quantile_preds, target_ci, chronos.quantiles.to(device))

                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(chronos.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            train_losses.append(loss.item())

            if (idx + 1) % 100 == 0:
                progress_bar.update(100)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        progress_bar.update(len(train_dataloader) % 100)
        progress_bar.close()
            
        average_training_loss = np.mean(train_losses)
        print(f"Epoch : {epoch+1}/{num_epochs} | Average Training Loss : {average_training_loss:.3f}")

    
    print("Chronos-2 Fine-Tuned Anomaly Detector Saved.")
    chronos.model.save_pretrained("./models/chronos-2")

    print("Evaluating :")
    anomaly_scores = []
    chronos.eval()

    with torch.inference_mode():
        for idx, batch in enumerate(test_dataloader):
            batch = batch.to(device)
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                batch_scores = chronos.anomaly_score(batch)
            anomaly_scores.extend(batch_scores.cpu().numpy())

    anomaly_scores = np.array(anomaly_scores)
    evaluate(anomaly_scores, test_labels, "Chronos-2 Base")

if __name__ == "__main__":
    main()