# Kaggle cells

These cells were run on Kaggle (Save & Run All, 2 x Tesla T4, Internet on) with the three datasets attached as inputs and the code
attached as a Kaggle dataset named `gc-ids`. The `*_setup.py` cells copy that dataset to `/kaggle/working/gc-ids`, install
`lightgbm`, `pyyaml` and (optionally) `pytorch-tabnet`, link the data (`scripts/link_kaggle_data.py`) and write the config files and
the UNSW-NB15 partition fix; **in this repository those config files (`configs/fast.yaml`, `configs/ext.yaml`,
`configs/extra_baselines.yaml`) and the fix (`gcids/data.py`) are already included**, so on a fresh Kaggle notebook it is enough to

```python
!git clone <REPOSITORY_URL> /kaggle/working/gc-ids
%cd /kaggle/working/gc-ids
!pip install -q lightgbm pyyaml && pip install -q --no-deps pytorch-tabnet
!python scripts/link_kaggle_data.py
```
and then to run the `*_run.py` cell of the experiment you want (`<kaggle-user>` in the cells is the owner of the Kaggle dataset).
Messages were translated to English; the logic is unchanged. Cells of commit 1 were not archived (see `REPRODUCE.md`).

| Cell | Produces |
|---|---|
| `commit2_run.py` | generator-quality dial, cGAN design variants (NSL-KDD), TabNet |
| `commit3_run.py` | UNSW-NB15 main table, S1 on NSL-KDD, TabNet on UNSW-NB15 |
| `commit4_run.py` | four extra seeds for the nine core variants, longer budget, C2-GAN-noAux on UNSW-NB15 |
| `commit4_fig3.py` | Fig. 3 (real versus generated samples) |

Note: in the original run of `commit4_run.py` the two Fig. 3 commands at the end of worker `w4` failed immediately (the Kaggle copy of the
code predated `gcids/figures_gen.py`); they did not affect any other result. Fig. 3 was produced afterwards in a separate session
that wrote `figures_gen.py` into the checkout and ran the same two commands as `commit4_fig3.py`.
