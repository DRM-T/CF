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

500 registered test alerts per dataset (five detector seeds × 100 alerts), one shared reconstructed oracle and budget.
CSR: share of alerts with a valid counterfactual; the other columns are over valid counterfactuals.

| Dataset | Method | CSR | changed features (mean) | L1 (median) | logit margin (median) |
|---|---|---|---|---|---|
| CIC-IDS2018 | CANDICE (tuned λ) | 0.908 | 17.46 | 14.84 | 0.46 |
| | DiCE | 1.000 | 16.38 | 22.20 | 3.20 |
| | **FlowSCE** | **1.000** | **1.07** | **6.04** | 1.18 |
| DoHBrw-2020 | CANDICE (tuned λ) | 0.866 | 19.82 | 26.85 | 0.20 |
| | DiCE | 0.030 | 18.27 | 19.39 | 1.31 |
| | **FlowSCE** | **1.000** | **1.29** | **17.57** | 1.13 |
| UNSW-NB15 | CANDICE (tuned λ) | 0.986 | 13.28 | 16.85 | 0.05 |
| | DiCE | 0.398 | 12.98 | 30.35 | 1.32 |
| | **FlowSCE** | **0.998** | **1.00** | **6.97** | 1.12 |

Full tables (detection, ablation, coupling, minimizers, bootstrap intervals, runtime) are in `results/SUMMARY.md` and
`results/paper/`.

## License

The code is released under the MIT License (see `LICENSE`). The dataset subsets in `data/` are derived from CIC-IDS2018,
DoHBrw-2020 and UNSW-NB15; the licensing and citation requirements of these original datasets apply to them.

## Citation

Please cite the paper if you use this code; the BibTeX entry will be added on publication.
