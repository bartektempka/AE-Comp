# Time Series Anomaly Detection with LSTM Autoencoders (AE, VAE, SAE)

This repository compares reconstruction-based anomaly detection methods on multivariate time series using LSTM Autoencoders: `AE_LSTM`, `VAE_LSTM`, and `SAE_LSTM`.

**Theoretical Introduction**
- **Reconstruction-based AD**: Train a model to reconstruct “normal” windows; anomalies are detected when reconstruction error increases. Out-of-distribution windows are not reconstructed well.
- **Reconstruction error**: For a window $x \in \mathbb{R}^{T\times F}$ and reconstruction $\hat{x}$, we use mean MSE over features:
	$$ e_t = \frac{1}{F} \sum_{f=1}^F (x_{t,f} - \hat{x}_{t,f})^2 $$

**Architecture: AE (LSTM Autoencoder)**
- **Encoder**: LSTM processes a window and returns hidden state `(h, c)` representing the sequence.
- **Decoder**: LSTM initialized with encoder state; zero input; reconstructs the window. Final `Linear(hidden_size → n_features)` with ReLU.
- **Loss**: MSE between reconstruction and input.
- Main code: [time_series_ad/AE_LSTM.py](time_series_ad/AE_LSTM.py)

Example reconstruction and loss (PyTorch Lightning):

```python
# AE: forward and training loss
def forward(self, x):
		_, hidden = self.encoder(x)
		decoder_input = torch.zeros(x.size(0), x.size(1), self.n_features).to(self.device)
		dec_out, hidden = self.decoder(decoder_input, hidden)
		return self.fc(self.relu(dec_out))

def training_step(self, batch, batch_idx):
		(x, _) = batch
		x_hat = self.forward(x)
		loss = nn.MSELoss()(x_hat, x)
		self.log("train_loss", loss)
		return loss
```

**Architecture: VAE (Variational LSTM Autoencoder)**
- **Encoder → latent**: LSTM → concatenate last `h_last` and `c_last` → `fc_mu`, `fc_logvar` → reparameterization $z = \mu + \sigma \odot \epsilon$.
- **Decoder**: Decoder hidden state initialized from $z$ (`fc_latent_to_hidden` for both `h` and `c`); zero input; projection to features like AE.
- **VAE Loss**: MSE + KL to standard normal $\mathcal{N}(0, I)$:
	$$ \mathcal{L} = \text{MSE}(\hat{x}, x) + \mathrm{KL}(q(z|x)\;\|\;p(z)) $$
	KL implementation:
	$$ \mathrm{KL} = -\frac{1}{2} \sum_{i=1}^d \big(1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2\big) $$
- Main code: [time_series_ad/VAE_LSTM.py](time_series_ad/VAE_LSTM.py)

Example latent and loss:

```python
# VAE: latent params and loss
_, (h, c) = self.encoder(x)
combined = torch.cat((h[-1], c[-1]), dim=1)
mu = self.fc_mu(combined)
logvar = self.fc_logvar(combined)
z = self.reparameterize(mu, logvar)
h_dec = self.fc_latent_to_hidden(z).unsqueeze(0)
c_dec = self.fc_latent_to_hidden(z).unsqueeze(0)
decoder_input = torch.zeros(x.size(0), x.size(1), self.n_features).to(self.device)
dec_out, _ = self.decoder(decoder_input, (h_dec, c_dec))
x_hat = self.fc(self.relu(dec_out))
loss = nn.MSELoss()(x_hat, x) + kl_divergence(mu, logvar)
```

**Architecture: SAE (Semi-Supervised LSTM Autoencoder)**
- **Goal**: Uses (partial) window labels to modulate the loss so normal samples are reconstructed well, while anomalies are “repelled”.
- **Weighted loss**: For labels $y \in \{0,1\}$:
	$$ \mathcal{L}(x, y) = (1-y)\cdot \mathrm{MSE}(\hat{x}, x) + y\cdot \lambda\, \mathrm{MSE}(\hat{x}, 1-x) $$
	where $\lambda$ increases penalty for anomalies (here: 10). This helps the model separate both modes.
- **Architecture**: Same as AE (LSTM encoder/decoder, projection to features).
- Main code: [time_series_ad/SAE_LSTM.py](time_series_ad/SAE_LSTM.py)

Loss fragment:

```python
# SAE: semi-supervised loss
(x, y) = batch
x_hat = self.forward(x)
loss = (1 - y).unsqueeze(-1) * nn.MSELoss(reduction="none")(x_hat, x) \
		 + y.unsqueeze(-1) * 10 * nn.MSELoss(reduction="none")(x_hat, 1 - x)
loss = loss.mean()
```

**Running Experiments**
- Requirements: Python 3.10+, PyTorch, PyTorch Lightning, NumPy, scikit-learn, pandas.
- Example to run experiment number 0 (AE, MSL dataset):

```bash
python time_series_ad/run_experiment.py --experiment_number 0
```

**Data**
- Uses multivariate timeseries from [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) benchmark datasets.
- Tested datasets: MSL, MITDB, GHL, SMD, SVDB, OPPORTUNITY, CATSv2, SMAP.
