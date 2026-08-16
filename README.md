# Transformer-Based-Anomaly-Detection-Using-Time-Series-Foundation-Models

### Introduction :

---

Anomaly detection in industrial time series is challengedby two persistent limitations: the cold-start problem, wherein conventional machine learning models require substantial historical data before deployment, and generalization failure, wherein models trained on one machine degrade severely on unseen hardware. This project investigates whether large pre-trained time series foundation models (TSFMs), fine-tuned via parameter-efficient Low-Rank Adaptation (LoRA), can overcome these limitations on the Server Machine Dataset (SMD).

---

### Baselines :

---

Isolation Forest<br>
One Class SVM<br>
LSTM Autoencoder<br>
PatchTST<br>

---

### Time Series Foundation Models :

---

MOMENT-1<br>
Chronos-2<br>

---

### Dataset :

---

[ServerMachineDataset](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)

Train : `./train/machine-1-1.txt`<br>
Test : `./test/machine-1-1.txt` (Same-Server/Machine)<br>
Test : `./test/machine-1-3.txt` (Cross-Server/Machine)<br>

---

### Fine-Tuning / Training Setup :

---

#### Baselines :

`window_size` : 100<br>

Isolation Forest -<br> 

`n_estimators`  : 100<br>
`contamination` : auto<br>
`max_samples`  : 256<br>

One Class SVM -<br> 

`nu` : 0.01<br>

LSTM Autoencoder -<br>

`num_epochs` : 10<br>
`learning_rate` : 1e-4<br>
`batch_size` : 128<br>
`hidden_dims` : 64<br>
`latent_dim` : 16<br>
`n_channels` : 38,<br>
`hidden_dims` : hidden_dims<br>
`latent_dim` : latent_dim<br>
`num_layers` : 2<br>
`dropout` : 0.1<br>
`optimizer` : AdamW<br>
`loss_fn` : MSE<br>
`anomaly_score` : MSE<br>

PatchTST -<br>

`num_epochs` : 10<br>
`learning_rate` : 1e-4<br>
`batch_size` : 128<br>
`patch_length` : 10<br>
`stride` : 10<br>
`d_model` : 128<br>
`d_ff` : 256<br>
`n_heads` : 4<br>
`n_layers` : 3<br>
`n_channels` : 38<br>
`optimizer` : AdamW<br>
`loss_fn` : MSE<br>
`anomaly_score` : MSE<br>

---

#### Time Series Foundation Models :

`window_size` : 96<br>

MOMENT-1 -<br>

`model` : MOMENT-1-base<br>
`num_epochs` : 7<br>
`learning_rate` : 5e-5<br>
`batch_size` : 16<br>
`n_channels` : 38<br>
`task` : "reconstruction"<br>
`r` : 8<br>
`lora_alpha` : 16<br>
`target_modules` : ["q", "v"]<br>
`lora_dropout` : 0.05<br>
`modules_to_save` : ["head"]<br>
`bias` : "none"<br>
`max_norm` : 1.<br>
`optimizer` : AdamW<br>
`scheduler` : OneCycleLR<br>
`loss_fn` : MSE<br>
`anomaly_score` : MSE<br>

Chronos-2 -<br>

`model` : chronos-2<br>
`num_epochs` : 7<br>
`learning_rate` : 5e-5<br>
`batch_size` : 16<br>
`n_channels` : 38<br>
`task` : "forecasting"<br>
`r` : 8<br>
`lora_alpha` : 16<br>
`target_modules` : ["q", "v"]<br>
`lora_dropout` : 0.05<br>
`modules_to_save` : ["head"]<br>
`bias` : "none"<br>
`forecasting_horizon` : 16<br>
`max_norm` : 1.0<br>
`optimizer` : AdamW<br>
`scheduler` : OneCycleLR<br>
`loss_fn` : Full Pinball Loss<br>
`anomaly_score` : CRPS<br>

---

### Experiments :

---

### Extended Cross-Server Evaluation :

Train : `machine-1-1.txt`<br>

Test :<br>

`machine-1-3.txt`<br>
`machine-1-4.txt`<br>
`machine-1-5.txt`<br>
`machine-1-6.txt`<br>
`machine-1-7.txt`<br>
`machine-1-8.txt`<br>

---

#### Few Shot Learning :

Few Shot Rates :<br>

0.1<br>
0.3<br>
0.7<br>

---

#### Sensor Dropout Rate :

Sensor Dropout Rates :<br>

0.1<br>
0.3<br>
0.7<br>

---

### Results :

---

#### Same-Server And Cross-Server Test :

![](./results/graph.png)

---

#### Extended Cross-Server Evaluation :

![](./results/extended_cross-server_evaluation.png)

---

#### Few-Shot Learning Test :

##### Same-Server :

![](./results/few_shot_same-server.png)

##### Cross-Server :

![](./results/few_shot_cross-server.png)

---

#### Sensor Dropout Test :

##### Same-Server :

![](./results/sensor_dropout_same-server.png)

##### Cross-Server :

![](./results/sensor_dropout_cross-server.png)

---

### Acknowledgement :

---

This project was developed in a Summer Internship at the EMC^2 Lab, Veermata Jijabai Technological Institute (VJTI), Mumbai, under the guidance of Prof. Madhavi Parimi and Dr. S.R. Wagh.
EMC Lab Website: https://emcc.in/

---

### License :

---

Distributed under the MIT License

---