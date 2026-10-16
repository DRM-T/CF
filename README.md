# FlowSCE: Graph-Consistent Sparse Counterfactual Explanations for Network Intrusion Detection

Code, data subsets and results of the paper *FlowSCE: Graph-Consistent Sparse Counterfactual Explanations for Network
Intrusion Detection* (submitted to *Expert Systems with Applications*).

A graph intrusion detector classifies a flow from its own attributes and from host summaries computed from the flows
(`X_V = D⁻¹ B X_E`). When a counterfactual changes the flow, these summaries change too. FlowSCE rebuilds them exactly for
every candidate, follows the total gradient through this update, and decides validity, coordinate restoration and a
margin-floored refinement (single-feature probes, repair, polishing) on the same reconstructed detector. The result is a
valid counterfactual that usually changes a single feature and keeps a decision margin of about ln 3.

## Installation

Tested with Python 3.11, PyTorch 2.2.2 + CUDA 12.1, on Ubuntu 22.04 with an NVIDIA RTX 5880 Ada. No graph library is needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Use the pinned versions: the registered split depends on pandas and scikit-learn, and the code stops if the recomputed split
differs from `results/splits/`.

## Data

`data/` holds the three subsets used in the paper (CIC-IDS2018 / NF-CICIDS2018-v3, DoHBrw-2020 layer 2, UNSW-NB15), with
row-level provenance and checksums (`data/README.md`). To rebuild them from the official archives:

```bash
python scripts/prepare_data.py --archives /path/to/downloads        # the three subsets
python scripts/prepare_doh_full.py --archives /path/to/downloads    # complete DoHBrw-2020 capture (coupling study only)
```

| Dataset | Official source |
|---|---|
| CIC-IDS2018 (NetFlow v3) | <https://espace.library.uq.edu.au/view/UQ:ece9b83> (`data_2018.zip`) |
| DoHBrw-2020 | <https://www.unb.ca/cic/datasets/dohbrw-2020.html> (`Total_CSVs.zip`) |
| UNSW-NB15 | <https://research.unsw.edu.au/projects/unsw-nb15-dataset> ("CSV Files") |

## Running

```bash
bash run_all.sh                                   # every experiment in order; finished steps are skipped
python scripts/make_paper_tables.py               # tables of the paper -> results/paper/
python scripts/audit.py                           # consistency checks of all results (non-zero exit on failure)
python tests/test_plus.py                         # guarantees of FlowSCE on validation alerts
python reference/run_reference_verification.py --out results/reference_verification   # constructed-input checks (seconds)
```

Explaining one alert in Python:

```python
import json
import numpy as np
import flowsce as L

st = L.prepare_dataset("cic", "data", "results", write=False)          # split, preprocessing, graphs
sp, dom, g = L.Space(st["X"]["train"], st["dom"]), st["dom"], st["G"]["test"]
L.POOL = L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
m = L.load_detector("results/ckpt/cic_tuned_11.pt", st["X"]["train"].shape[1])
cfg = L.ExplainCfg(**json.load(open("results/protocol/explain_config.json"))["cic"])
plus = json.load(open("results/protocol/plus_config.json"))["cic"]
k = int(np.where(L.predict(m, g) >= 0.5)[0][0])                         # first flow predicted as attack
row = L.explain_plus(m, g, k, sp, dom, cfg, **L.plus_kwargs(plus))
print(row["v"], row["support_eps"], row["margin_logit"])                 # valid?, changed features, logit margin
```

All settings (detector, explainer, refinement, baselines) were selected on validation data and are stored in
`results/protocol/`. GPU scatter operations are not bitwise deterministic, so repeated runs can differ in rare tie-breaks.

## Repository layout

| Path | Content |
|---|---|
| `flowsce/` | package: data and split, schema and validator, graph construction, detector, reconstructed oracle, FlowSCE (`explain.py`), baselines (CANDICE, DiCE via `dice-ml`, LBS-adapted, RHC), metrics, bootstrap, batched GPU path (`fast.py`) |
| `scripts/` | one script per experiment (called by `run_all.sh`) |
| `tests/` | FlowSCE guarantees; sequential vs batched minimizers |
| `reference/` | specification-level reference implementation and its verification on constructed inputs (CPU, float64): rank-one update vs independent rebuild, excluded-reference invariance, total gradient vs finite differences, restoration invariants, contract tests |
| `data/` | dataset subsets, provenance, checksums |
| `results/` | checkpoints, locked configurations, per-alert ledgers, summaries, paper tables, audit report |

## Main results

Test partitions; five detector seeds; 500 alerts registered before explanation per dataset (five seeds × 100). All settings
were selected on validation data. Every explainer queries the same reconstructed oracle with the same budget (4096 forward
evaluations per alert). Values are listed as CIC-IDS2018 / DoHBrw-2020 / UNSW-NB15.

**Detection** (test macro-F1, mean of five seeds; `results/paper/tab_T3_detection.tex`)

| Model | CIC-IDS2018 | DoHBrw-2020 | UNSW-NB15 |
|---|---|---|---|
| **FlowSCE detector** (explained) | **0.9999** | 0.9990 | **0.9926** |
| Edge-only (no host context) | 0.9991 | 0.9935 | 0.9926 |
| E-GraphSAGE | 0.9999 | **0.9998** | 0.9654 |
| CANDICE detector (β-TCVAE + RF) | 0.9996 | 0.9703 | 0.9911 |

**Generators** (one counterfactual per alert; `tab_T6_generators.tex`). CSR: share of alerts with a valid counterfactual;
the other columns are over valid counterfactuals.

| Method | CSR | changed features (mean) | L1 (median) | logit margin (median) |
|---|---|---|---|---|
| CANDICE (tuned λ) | 0.908 / 0.866 / 0.986 | 17.46 / 19.82 / 13.28 | 14.84 / 26.85 / 16.84 | 0.46 / 0.20 / 0.05 |
| DiCE (dice-ml) | 1.000 / 0.030 / 0.398 | 16.38 / 18.27 / 12.98 | 22.20 / 19.38 / 30.35 | 3.20 / 1.31 / 1.32 |
| FlowSCE search only (A0) | 1.000 / 1.000 / 0.998 | 18.53 / 19.83 / 13.37 | 20.74 / 38.69 / 19.52 | 1.45 / 0.78 / 0.23 |
| **FlowSCE** (search + refinement) | **1.000 / 1.000 / 0.998** | **1.07 / 1.29 / 1.00** | **6.04 / 17.57 / 6.97** | 1.18 / 1.13 / 1.12 |

**Minimizers on the same search outputs** (A0 seeds; `tab_T7_minimizers.tex`, paired intervals in `tab_A1_bootstrap.tex`).
Oracle calls are those of the minimizer itself (median).

| Minimizer | changed features (mean) | L1 (median) | logit margin (median) | oracle calls |
|---|---|---|---|---|
| none (search output) | 18.53 / 19.83 / 13.37 | 20.74 / 38.69 / 19.52 | 1.45 / 0.78 / 0.23 | 0 |
| DiCE post-hoc sparsity | 17.18 / 19.64 / 12.66 | 18.82 / 37.94 / 19.10 | 0.41 / 0.56 / 0.15 | 12 / 1 / 6 |
| Restoration (one pass) | 6.77 / 10.03 / 5.36 | 10.69 / 26.15 / 10.89 | 0.28 / 0.28 / 0.20 | 19 / 20 / 14 |
| RHC | 1.06 / 1.29 / 1.01 | 11.94 / 28.14 / 29.21 | 3.16 / 2.48 / 2.33 | 121 / 149 / 115 |
| LBS-adapted | 1.05 / 1.37 / 1.02 | 8.09 / 21.09 / 8.24 | 1.34 / 1.01 / 0.72 | 121 / 149 / 115 |
| **FlowSCE refinement** | 1.07 / 1.29 / 1.00 | **6.04 / 17.57 / 6.97** | 1.18 / 1.13 / 1.12 | 1111 / 1543 / 825 |

FlowSCE and LBS-adapted reach about one changed feature; FlowSCE stays closer to the alert on CIC-IDS2018 and DoHBrw-2020,
at about ten times the oracle calls. At a matched cost (refinement without probes, `tab_A9_ablation.tex`) and within the
1st–99th training percentiles (`tab_A11_constrained.tex`), LBS-adapted is as sparse or sparser. Stale host summaries make
a counterfactual spuriously valid in 0 of 2776 nominal successes on alerts with coupling ≤ 0.01, but in 13.8% of those on alerts with coupling above 0.5 (`tab_A10_coupling.tex`).

Full tables (ablation, coupling, bootstrap intervals, plausibility, runtime) are in `results/SUMMARY.md` and `results/paper/`.

## License

The code is released under the MIT License (see `LICENSE`). The dataset subsets in `data/` are derived from CIC-IDS2018,
DoHBrw-2020 and UNSW-NB15; the licensing and citation requirements of these original datasets apply to them.

## Citation

Please cite the paper if you use this code; the BibTeX entry will be added on publication.
