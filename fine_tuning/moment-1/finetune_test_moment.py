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
from momentfm import MOMENTPipeline
from peft import LoraConfig, get_peft_model

# MOMENT-1 Fine-Tuned Anomaly Detector

class MOMENTFineTunedAnomalyDetector(nn.Module):

    def __init__(self, n_channels):

        super().__init__()

        # Pretrained Time Series Foundation Model

        self.pretrained_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-base",
            model_kwargs={"task_name": "reconstruction", "use_reentrant": True},
        )
        self.pretrained_model.init()

        # LoRA Configuration Parameters
    
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            modules_to_save=["head"],
            bias="none"
        )

        # Model

        self.model = get_peft_model(self.pretrained_model, self.lora_config)

        self.hidden_dims = self.pretrained_model.config.d_model

    def anomaly_score(self, x):

        B, T, C = x.shape
        
        # Convert to Channel Independent shape: (B*C, 1, T)
        x_ci = x.permute(0, 2, 1).reshape(B * C, 1, T)
        
        # Forward pass independently
        reconstructed_ci = self.forward(x_ci) 
        
        # Reshape back to original multivariate shape: (B, T, C)
        reconstructed = reconstructed_ci.squeeze(1).reshape(B, C, T).permute(0, 2, 1)
        
        # Calculate Squared Error
        mse = ((x - reconstructed)**2)
        
        mse_per_channel = mse.mean(dim=1)
        anomaly_score = mse_per_channel.mean(dim=1)
        
        return anomaly_score

    def forward(self, x):

        # (B*C, 1, T)

        outputs = self.model(x_enc=x)
        reconstructed = outputs.reconstruction
        return reconstructed

def evaluate(anomaly_scores, labels, algo):

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
    args = parser.parse_args()

    window_size = args.window_size
    num_epochs = 7
    learning_rate = 5e-5
    batch_size = 16

    train_path = "./ServerMachineDataset/train/machine-1-1.txt"
    test_path = "./ServerMachineDataset/test/machine-1-1.txt"
    test_label_path = "./ServerMachineDataset/test_label/machine-1-1.txt"

    train_data, test_data, test_label_data = preprocess(train_path, test_path, test_label_path)
    test_labels = test_label_data[window_size - 1 :]

    train_dataloader, test_dataloader = get_dataloaders(
        train_data, test_data, window_size, 
        num_workers=2, pin_memory=True, batch_size=batch_size, is_channel_independent=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device : {device}")

    moment = MOMENTFineTunedAnomalyDetector(n_channels=38).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(params=moment.parameters(), lr=1e-6)
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
            
            batch_ci = batch.permute(0, 2, 1).reshape(B * C, 1, T)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                reconstructed_ci = moment(batch_ci)
                loss = loss_fn(reconstructed_ci, batch_ci)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(moment.parameters(), max_norm=1.0)
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

    
    print("MOMENT-1 Fine-Tuned Anomaly Detector Saved.")
    moment.model.save_pretrained("./models/moment-1")

    print("Evaluating :")
    anomaly_scores = []
    moment.eval()

    with torch.inference_mode():
        for idx, batch in enumerate(test_dataloader):
            batch = batch.to(device)
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                batch_scores = moment.anomaly_score(batch)
            anomaly_scores.extend(batch_scores.cpu().numpy())

    anomaly_scores = np.array(anomaly_scores)
    evaluate(anomaly_scores, test_labels, "MOMENT-1 Base")

if __name__ == "__main__":
    main()