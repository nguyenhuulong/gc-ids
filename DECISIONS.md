# DECISIONS (design choices that the design notes of the study did not fix)

Format: what, why, alternatives. "SPEC" in code comments refers to the internal design notes of this re-implementation.
None of these choices was made after looking at test-set results. Results of the manuscript use `configs/fast.yaml` (reduced budget); the epoch counts below are the defaults of `configs/base.yaml`.

| # | Decision | Why | Alternatives |
|---|---|---|---|
| D1 | The code of the first submission was not preserved; this repository is a clean re-implementation guided by the authors' design notes. All numbers in the revised manuscript come from this code and replace those of the first submission. | The original code could not be recovered. | – |
| D2 | One-hot + min-max are fit on the 80% TRAIN part only (after the seeded 80/20 train/val split); val/test are clipped to [0,1]. | No leakage: the scaler and encoder never see validation or test data. | Fit on whole official train (small leak into val). |
| D3 | Encoder = Linear-BN-ReLU-Dropout(0.3) x2 (512, 256) then Linear to 128-d (no activation on the last layer). Contrastive embedding = L2-normalised output. | "MLP [512,256,128]" in the notes; a ReLU on the last layer would confine embeddings to the positive orthant and distort uniformity. | ReLU on last layer. |
| D4 | S0/S1 embeddings for the embedding metrics = L2-normalised 128-d encoder output (penultimate layer). | Same space as the contrastive encoders. | Raw output. |
| D5 | cVAE loss = summed squared error + beta*KL (summed over latent), beta=4 with linear warm-up over 20% epochs. | Mean-MSE with sum-KL collapses the posterior. | Mean-MSE with smaller beta. |
| D6 | cGAN: WGAN-GP + ACGAN auxiliary classifier in D (`aux_classifier: true`, weight 1, G also receives the classifier loss). This is how the generator of the study was described in the design notes; the first submission did not state it. Ablation `ablation_gan.yaml` runs it off. | Honest description of the label channel; second label-leak control. | Plain WGAN-GP. |
| D7 | cGAN fine-tune scope default = `all` (paper text). Design notes say "minority"; both are run in `ablation_gan.yaml` (`finetune_scope: minority`). | The two sources contradict each other. A pure-minority phase 2 makes the generator forget majority classes, which would be a bug-like side effect, so it is tested, not assumed. | Default minority. |
| D8 | Class-aware sampling for generator training: class probability proportional to N_k^0.5. | Rare classes must be learned. | Plain sampling. |
| D9 | Mixup partner = random REAL sample of the same class, lambda ~ Beta(0.2,0.2) then max(lambda,1-lambda), view1 = x. | Notes say "alpha = 0.2". | lambda ~ U(0.7,1). |
| D10 | Random mask = zero-out 30% of features independently per view. | Notes. | Mean-fill. |
| D11 | S1 rule A = raise every class to the median class count (added samples capped at 50 x N_k); rule B = `ratio` x N_train samples spread uniformly (original style). Both reported, sweep in `ablation_s1.yaml`. | Original rule is only known as "synth_ratio = 1.0". | – |
| D12 | Probe = MLP 128-64-K (main) and linear 128-K; class-weighted CE w_k = N/(K N_k); Adam 1e-3, 50 epochs, batch 256; the epoch with the best VALIDATION Macro-F1 is kept. | SPEC. | – |
| D13 | Alignment: mean squared distance of random same-class pairs, LOWER is better (the first submission marked it as higher-is-better by mistake). | Definition from Wang & Isola; notes call it a within-class distance proxy. | – |
| D14 | MMD: unbiased RBF, bandwidth = median heuristic x {0.5,1,2,4}, reported as the mean over the four. Compare the order cVAE vs cGAN rather than absolute values. | Kernel bandwidth is a free choice. | – |
| D15 | CIC-IDS2017: classes >= 2000 rows, cap 50,000 per class, fixed stratified 70/30 split (seed 0); log1p for non-negative columns with train max > 1e3 (columns chosen on train). | Heavy-tailed flow features. | No log1p. |
| D16 | Embedding metrics use a fixed-seed stratified subsample of at most `eval.n_eval` (20,000) TEST points; kNN (k=20) uses a train subsample of the same size. | Silhouette is O(n^2). | Full test set. |
| D17 | Pretraining uses Adam (no weight decay), cosine LR over epochs, batch 256, tau 0.5, drop-last batches. | SPEC. | AdamW. |
| D18 | Tuning (`gcids.tune`) picks ONE shared (tau, lr, sigma) for all contrastive methods on validation data of a tuning seed outside the final seed list. Not run by default; defaults are the SPEC values for every method. | Fairness. | Per-method tuning. |
| D19 | The M2M-VAEGAN code of Kang et al. is NOT included (no licence file); an optional third generator can be added behind the same `GenWrapper.sample(y)` interface. | Licence. | Vendoring the code. |

## Known limitations
- Reduced training budget (`configs/fast.yaml`) for all reported results; a longer budget was tested only for S0 and C2-GAN on UNSW-NB15 (`configs/ext.yaml`).
- Architecture details of the generators (hidden sizes, epochs, beta) are choices of this re-implementation; every result line stores its `cfg_hash`.
- GPU training is not bit-wise reproducible (see README, "Reproducibility notes").
- CIC-IDS2017 loading is implemented (D15) but was not evaluated in the manuscript.
