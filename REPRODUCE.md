# Which command produced which file

All runs used `configs/fast.yaml` (reduced budget) on Kaggle (2 x Tesla T4, several processes per GPU) and wrote their
raw lines to `results_archive/<commit>/raw/`. "commit" is the Kaggle *Save & Run All* session that produced the files. Each
run is resumable. `CFG` below stands for `--config configs/base.yaml configs/<dataset>.yaml configs/fast.yaml`.

Seeds: `SEEDS6 = 42 123 456 789 1011 2024` (main tables); extra seeds `2025 3407 7777 31337` (ten-seed comparisons).

| Archive file | Seeds | Variants | Command (abridged) |
|---|---|---|---|
| `commit1/raw/nslkdd_s{0,1}.jsonl` | SEEDS6 (split by `--shard`) | S0 C1 C1b-noise C1b-mixup C2-VAE C2-GAN SupCon C3-realpair GC-mask-GAN GC-mask-VAE SCARF LightGBM FT-Transformer GenQ-VAE GenQ-GAN | `python -m gcids.run --dataset nslkdd CFG --seeds SEEDS6 --variants ... --shard i/2 --out results/raw/nslkdd_s{i}.jsonl` ¹ |
| `commit2/raw/dial_nslkdd.jsonl` | 42 123 456 | generator-quality dial (cGAN 1k/2k/4k/8k iterations, cVAE 10/25/50/100 epochs) | `... configs/ablation_dial_fast.yaml configs/fast.yaml --seeds 42 123 456 --out results/raw/dial_nslkdd.jsonl` |
| `commit2/raw/gan_nslkdd.jsonl` | 42 123 456 | C2-GAN, C2-GAN-noAux, C2-GAN-minority, GenQ-GAN, GenQ-GAN-noAux, GenQ-GAN-minority | `... configs/ablation_gan.yaml configs/fast.yaml --seeds 456 123 42 --variants <those> --out results/raw/gan_nslkdd.jsonl` |
| `commit2/raw/tabnet_nslkdd.jsonl` | SEEDS6 | TabNet | `... configs/extra_baselines.yaml configs/fast.yaml --variants TabNet --out results/raw/tabnet_nslkdd.jsonl` |
| `commit3/raw/unswnb15_s{0,1}.jsonl` | SEEDS6 (split by `--shard`) | the 16 variants of commit 1 plus S1 | `python -m gcids.run --dataset unswnb15 CFG --seeds SEEDS6 --variants S0 S1 C1 ... GenQ-GAN --shard i/2 --out results/raw/unswnb15_s{i}.jsonl` |
| `commit3/raw/nslkdd_S1.jsonl` | SEEDS6 | S1 | `... --dataset nslkdd CFG --variants S1 --out results/raw/nslkdd_S1.jsonl` |
| `commit3/raw/tabnet_unswnb15.jsonl` | SEEDS6 | TabNet | as for NSL-KDD, `--dataset unswnb15` |
| `commit4/raw/ext_{nsl,unsw}_{a,b}.jsonl` | 2025 3407 (a), 7777 31337 (b) | S0 C1 C1b-noise SupCon C3-realpair SCARF C2-VAE C2-GAN GC-mask-GAN | `... CFG configs/ext.yaml --seeds <two seeds> --variants <those> --out results/raw/ext_<ds>_<a|b>.jsonl` |
| `commit4/raw/unsw_long_noaux.jsonl` | 42 123 456 | S0-long C2-GAN-long C2-GAN C2-GAN-noAux GenQ-GAN GenQ-GAN-noAux | `--dataset unswnb15 CFG configs/ext.yaml --seeds 42 123 456 --variants ...` |
| `commit4/raw/unsw_long_b.jsonl` | 789 1011 2024 | S0-long C2-GAN-long | `--dataset unswnb15 CFG configs/ext.yaml --seeds 789 1011 2024 --variants S0-long C2-GAN-long` |
| `commit4/figures/realgen_*` | 42 | cVAE and cGAN retrained for the figure | `python -m gcids.figures_gen --dataset <ds> --config configs/base.yaml configs/<ds>.yaml configs/fast.yaml --seed 42 --cache cache --results results --out results/figures --allow-train --device cuda` |

¹ The notebook cells of commit 1 were not archived; the command above is reconstructed from the shard structure of the result
files (seeds 42, 456, 1011 in `_s0` and 123, 789, 2024 in `_s1`). The cells of commits 2 to 4 are in `kaggle/`.

The generators (cVAE, cGAN) are trained once per (dataset, seed) and cached in `cache/gen/`; all variants of a seed share them.
Some result files overlap in (dataset, seed, variant) by design (for example C2-GAN appears in commit 3 and in commit 4's
`unsw_long_noaux.jsonl`): the second occurrence is an independent re-run used for the same-run comparison with C2-GAN-noAux and
for the estimate of run-to-run variation. `analysis/` handles these overlaps explicitly (one file per key in each analysis).
