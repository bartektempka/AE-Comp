# Semi-Supervised Anomaly Detection via Variational Autoencoders

## 1. Introduction

Anomaly Detection (AD) is the identification of rare items, events, or observations which raise suspicions by differing significantly from the majority of the data. This project focuses on visual anomaly detection in industrial contexts using **Deep Generative Models**.

Data set used in the tests can be found at [MVTec AD Data set](https://www.mvtec.com/company/research/datasets/mvtec-ad). This data set should be downloaded and placed at the `root` of the repository (`/mvtec_anomaly_detection`).

The `evaluate_all_models.ipynb` can be run to get the results for comparison of all models across all categories from the dataset.

The `ecaluation_results_all_models.csv` contains results from models saved incrementaly. If you wish to retrain the models again remove that file for hte results to be saved again.

### The Core Hypothesis
We operate under the **Reconstruction-Based Anomaly Detection** paradigm. The core assumption is that an autoencoder trained strictly on a "normal" manifold $\mathcal{M}$ will generalize well within that manifold but fail to generalize to samples outside of it (anomalies).

Mathematically, for an input $x$:
$$
\text{AnomalyScore}(x) = || x - \hat{x} ||^2
$$
Where $\hat{x}$ is the reconstruction. Anomalies yield high residuals because they cannot be effectively compressed into the latent space learned from normal data.

### Comparison of VAE and VAE-GRF

![VAE vs VAE-GRF](./images/vae-vae-grf.png)


---

## 2. Methodology

In the project we train the data on the dataset that contains only "good" examples, from that the models learn how to reproduce images and when encountering "defective" example we expect the reproduction error to be large allowing to classify it as an anomaly

### A. Training Phase (Learning Normality)
* **Input:** The model observes set $X_{train}$ consisting **only** of non-defective ("Good") samples.
* **Objective:** Maximize the likelihood $P(X)$ of the normal data.
* **Latent Space:** The model maps these images to a compressed latent space $z$. The goal is to force the model to learn the *features* of normality (e.g., the texture of wood, the curve of a bottle) rather than memorizing pixels.

### B. Inference Phase (Detecting Defects)
* **Input:** The model is evaluated on $X_{test}$, which contains both "Good" and "Defective" samples.
* **Scoring:** We calculate the pixel-wise reconstruction error.
* **Decision Boundary:** A threshold $\tau$ is applied to the error.
    * If $\text{Error} > \tau \rightarrow$ **Anomaly**
    * If $\text{Error} \leq \tau \rightarrow$ **Normal**

### Dataset: MVTec AD
We utilize the **MVTec Anomaly Detection Dataset**, a challenging benchmark covering 15 categories. It tests two distinct types of visual processing:
1.  **Textures:** Statistical, repetitive patterns (e.g., Carpet, Wood, Grid).
2.  **Objects:** Structural, centered objects (e.g., Bottle, Transistor, Screw).

### Visual Examples
The distinction is often subtle or structural.

| **Training Sample (Good)** | **Test Sample (Defective)** |
|:---:|:---:|
| ![Good Bottle](../mvtec_anomaly_detection/bottle/train/good/000.png) | ![Broken Bottle](../mvtec_anomaly_detection/bottle/test/broken_large/000.png) |
| *Model learns this structure* | *Model fails to reconstruct the break* |

---

## 3. Model Architectures

We investigated four architectures, moving from deterministic compression to probabilistic generative modeling.

### A. Convolutional Autoencoder (CAE)
* **Mechanism:** A deterministic bottleneck. The encoder consists of Convolutional blocks (Conv2D + BatchNorm + ReLU) that downsample spatial dimensions while increasing channel depth.
* **The Bottleneck:** The latent vector $z$ is a dense, compressed representation of fixed size.
* **Loss Function:** Standard Mean Squared Error (MSE).
    $$\mathcal{L}_{CAE} = || x - Dec(Enc(x)) ||^2$$
* **Pros/Cons:** Simple to train, but the latent space is often discontinuous. This means interpolation between two valid "normal" images might result in nonsense, limiting its generative capability.

### B. Variational Autoencoder (VAE)
* **Mechanism:** The VAE introduces a probabilistic twist. Instead of mapping input $x$ to a single point $z$, the encoder predicts the parameters (mean $\mu$ and variance $\sigma^2$) of a probability distribution $q(z|x)$.

* Still should train rather quickly but provide the possiblity of better results than basic CAE.

### C. Vision Transformer VAE (ViT-VAE)
* **Mechanism:** Replaces the CNN backbone with a Transformer. The image is split into fixed-size patches (e.g., 16x16 pixels), linearly embedded, and processed via **Self-Attention**.
* **The "Global Context" Hypothesis:** CNNs have a limited "receptive field"—they only see local neighbors. If a bottle is missing its cap, a CNN might not realize the cap is missing if it's looking at the bottom of the bottle. A Transformer sees all patches simultaneously.


### D. VAE with Gaussian Random Field (VAE-GRF)
* **Mechanism:** Standard VAEs flatten the spatial dimensions into a 1D vector, destroying spatial correlation. VAE-GRF maintains a **2D latent map**.
* **Spatial Prior:** Instead of assuming independence in the latent space, it models the latent variable using a Gaussian Random Field. This enforces that neighboring points in the latent map should be similar.
* **Use Case:** Ideal for **Textures**. A scratch on wood is an interruption of a continuous pattern. The GRF prior strictly penalizes high-frequency disruptions in the latent representation.

### Example Gaussian Random Field
![GRF](./images/grf.png)

---

## 4. Evaluation and Results

Models were evaluated using **ROC AUC** (Receiver Operating Characteristic - Area Under Curve), which measures the model's ability to rank anomalous images higher than normal images, independent of a specific threshold.

### Key Findings

#### 1. The "Texture vs. Object" Trade-off
The results highlight a fundamental dichotomy in anomaly detection:
* **Texture-centric models (VAE-GRF):** Excel at detecting interruptions in continuous patterns (Wood, Carpet) because they enforce spatial smoothness. However, they fail on objects like "Screw" because screws *have* high-frequency edges that are essentially "normal."
* **Structure-centric models (ViT-VAE):** The Transformer performed poorly on average but excelled on "Bottle" and "Transistor." Its global attention mechanism successfully modeled the holistic shape of the object, detecting large structural defects that CNNs missed.

#### 2. The Robustness of the Baseline VAE
Despite the theoretical advantages of Transformers and GRFs, the standard VAE offered the best mean performance. This suggests that without massive datasets (Transformers are notoriously data-hungry), the inductive bias of a standard Convolutional VAE strikes the best balance for general-purpose anomaly detection on small-scale industrial data.

### Visualization of Results

#### 1. F1 Scores
![F1 scores comparison](./results/f1_score_comparison.png)

#### 2. ROC AUC Comparison across Categories
![ROC AUC Comparison](results/roc_auc_comparison.png)

#### 3. Heatmap of Model Performance
The heatmap reveals that **Wood** was the easiest category for all models, while **Screw** and **Metal Nut** were universally difficult, suggesting current reconstruction-based methods struggle with rotational/logical anomalies.
![ROC AUC Heatmap](results/roc_auc_heatmap.png)

#### 4. Metric Distributions
![Metrics Distribution](results/metrics_distribution.png)

---

## 5. Summary 
We successfully implemented a semi-supervised anomaly detection pipeline. While specialized architectures (ViT, GRF) offer benefits for specific data types, the standard VAE remains the most robust baseline.