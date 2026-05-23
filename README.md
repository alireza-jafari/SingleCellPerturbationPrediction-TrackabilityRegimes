<h1 align="center">Single-Cell Perturbation Prediction and Trackability Regimes</h1>

<p align="center">
  <b>Code release for the ICML 2026 Spotlight Position Paper</b><br>
  <i>Position: Temporal Measurement Interval Determines Computational and Model Complexity in Single-Cell Perturbation Analysis</i>
</p>

<p align="center">
  <a href="https://github.com/alireza-jafari/SingleCellPerturbationPrediction-TrackabilityRegimes/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-supported-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/ICML-2026%20Spotlight-purple" alt="ICML 2026 Spotlight">
  <img src="https://img.shields.io/badge/Single--Cell-Perturbation%20Prediction-teal" alt="Single-cell perturbation prediction">
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=lECKpTE1lW">OpenReview</a> ·
  <a href="https://icml.cc/virtual/2026/poster/67090">ICML Poster</a> ·
  <a href="https://github.com/alireza-jafari/SingleCellPerturbationPrediction-TrackabilityRegimes">Repository</a>
</p>

<p align="center">
  <img src="assets/overview_trackability.png" width="920" alt="Overview of trackable and untrackable regimes for single-cell perturbation prediction">
</p>

---

## Core message

Single-cell perturbation prediction is usually treated as a modeling problem: collect unpaired control and treated cell populations, then use a sufficiently expressive model to map one distribution to the other. This repository supports a different conclusion:

> **The measurement time gap is an experimental knob that controls both computational tractability and model complexity.**

When the post-perturbation measurement is collected before a critical time gap $\Delta$, the latent correspondence between control and treated populations can be recovered efficiently under the paper's assumptions. After this correspondence is recovered, learning the transition map reduces to supervised estimation, and a simple linear model can be sufficient. When the measurement gap exceeds $\Delta$, correspondence recovery becomes NP-hard in the worst case, even when the underlying transition is linear.

---

## Problem formulation

We observe two unpaired populations:

- control cells at time $0$: $X^{(0)} = [x_1^{(0)}, \dots, x_n^{(0)}] \in \mathbb{R}^{d \times n}$,
- treated cells at time $t$: $X^{(t)} = [x_1^{(t)}, \dots, x_n^{(t)}] \in \mathbb{R}^{d \times n}$.

The paper models the treated population as

$$
x_i^{(t)} = F_t\left(x_{\sigma(i)}^{(0)}\right),
$$

where $F_t$ is the perturbation transition map and $\sigma$ is an unknown latent matching between populations.

The critical time gap is

$$
\Delta = \sqrt{\frac{1-\delta}{2nL^2}},
$$

where $L$ controls temporal smoothness of the transition map and $\delta$ is the restricted-isometry constant of the initial cell-state matrix.

| Regime | Condition | Computational consequence | Modeling consequence |
|---|---:|---|---|
| **Trackable** | $t < \Delta$ | permutation/coupling recovery is polynomial-time tractable | after matching, the task reduces to supervised learning |
| **Untrackable** | $t > \Delta$ | recovery is NP-hard in the worst case | expressive nonlinear models do not remove the intrinsic barrier |

---

## Method: Linear Alternating Optimal Transport (LAOT)

LAOT is intentionally minimal. It alternates between a correspondence step and a linear-map step:

```text
Input: control matrix X0, treated matrix Xt, number of iterations K
Initialize W = I

for k = 1, ..., K:
    1. Matching step:
       find Π that minimizes ||Xt - W X0 Π||_F^2
       using a linear assignment / optimal transport solver

    2. Linear map step:
       find W that minimizes ||Xt - W X0 Π||_F^2
       using ordinary least squares

Output: W and predictor x_new^(t) = W x_new^(0)
```

The method is not introduced as a high-capacity model. Its purpose is to isolate the role of measurement timing: when the experimental design places the task in the trackable regime, a simple linear transition can match or outperform larger nonlinear baselines.

---

## Visual summary of the main empirical result

### Synthetic phase transition

<p align="center">
  <img src="assets/synthetic_phase_transition.png" width="550" alt="Synthetic phase transition in permutation recovery">
</p>

In the controlled synthetic setup, the ground-truth map follows a near-identity linear path $W_\star(t)=I+tE$. LAOT recovers the latent permutation with near-perfect accuracy for small $t$, but recovery rapidly collapses after the transition into the NP-hard regime. Larger dimensions expand the trackable window.

### Nonlinear baselines also degrade beyond trackability

<p align="center">
  <img src="assets/untrackable_model_degradation.png" width="950" alt="Model degradation beyond the trackable regime for Compact CellOT, scGen, and LAOT">
</p>

The collapse is not specific to LAOT. Compact CellOT, scGen, and LAOT all show increasing distributional error once the measurement gap enters the untrackable regime.

### Optimization stability in the trackable regime

<p align="center">
  <img src="assets/optimization_stability.png" width="950" alt="Optimization stability comparison on AP-1 COLO858 DMSO to VEM">
</p>

On the within-context AP-1 task, high-capacity CellOT can show non-monotone training dynamics, Compact CellOT stabilizes part of the trajectory, and LAOT converges rapidly with nearly coincident train/test curves.

---

## Benchmarks reproduced in this repository

This code release supports the experiments in the paper across synthetic and biological perturbation settings.

| Benchmark | Modality | Setting | Main purpose |
|---|---|---|---|
| Synthetic near-identity data | simulated vectors | varying $t$ and $d$ | phase transition in latent permutation recovery |
| AP-1 | targeted protein panel | DMSO $\rightarrow$ VEM, 48h | within-cell-line and replicate perturbation prediction |
| 4i | multiplexed protein imaging | drug exposure, 8h | within-context drug perturbation prediction |
| SciPlex3 | scRNA-seq | 24h drug response | transcriptome-scale perturbation prediction |
| 2i time course | scRNA-seq trajectory | 12h to 168h horizons | biological time-gap sweep |
| Cross-cell-line AP-1 | targeted protein panel | held-out cell line | out-of-context generalization stress test |

### Representative paper results

Lower MMD$^2$ is better.

| Dataset | Condition | CellOT | scGen | Compact CellOT | LAOT |
|---|---:|---:|---:|---:|---:|
| AP-1 | COLO858 | 0.0995 | 0.0172 | 0.0019 | **0.0006** |
| AP-1 | WM902B | 0.0443 | 0.1423 | 0.0015 | **0.0007** |
| AP-1 | SKMEL19 | 0.1122 | 0.0323 | 0.0016 | **0.0011** |
| 4i | Imatinib | 0.0700 | 0.0330 | 0.0079 | **0.0063** |
| 4i | Trametinib | 0.0463 | 0.0098 | **0.0076** | 0.0080 |
| 4i | Dexamethasone | 0.0685 | 0.0160 | 0.0075 | **0.0071** |

| SciPlex3 drug | CellOT | scGen | Compact CellOT | LAOT |
|---|---:|---:|---:|---:|
| Trametinib | 0.0078 | 0.0059 | 0.0048 | **0.0040** |
| Givinostat | 0.0117 | 0.0083 | 0.0079 | **0.0033** |
| Abexinostat | 0.0129 | 0.0091 | 0.0074 | **0.0038** |

---

## Repository structure

The current release is notebook-centered so that each major paper result can be inspected and rerun directly.

```text
.
├── README.md
├── LICENSE
├── assets/
│   ├── overview_trackability.png
│   ├── synthetic_phase_transition.png
│   ├── untrackable_model_degradation.png
│   └── optimization_stability.png
│
├── LAOT_Synthetic_data.ipynb
├── Cellot_Synthetic_data.ipynb
├── scGen_Synthetic_data.ipynb
│
├── AOT_AP-1_in_a_drug.ipynb
├── AOT_AP-1_in_a_drug_replicate.ipynb
├── CellOT_AP-1_in_a_drug.ipynb
├── identity_Heman_in_a_drug.ipynb
│
├── AOT_4I_in_a_drug.ipynb
├── CellOT_4I_in_a_drug.ipynb
├── identity_4I_in_a_drug.ipynb
│
├── AOT_2I_in_a_drug.ipynb
│
├── AOT_single_cell_in_a_drug.ipynb
├── csGen_single_cell_in_a_drug.ipynb
├── CellOT_scGen_single_cell_in_a_drug.ipynb
├── Small_CellOT_scGen_single_cell_in_a_drug.ipynb
└── identity_single_cell_in_a_drug.ipynb
```

> **Naming note.** Some notebooks use `AOT` in the filename. These notebooks implement the alternating optimal transport procedure used as LAOT in the paper.

---

## Installation

A minimal environment can be created with Conda:

```bash
conda create -n trackability python=3.10 -y
conda activate trackability

pip install numpy scipy pandas scikit-learn matplotlib seaborn tqdm jupyter ipykernel
pip install torch scanpy anndata pot
```

Some notebooks import CellOT-specific modules. Install CellOT following the official CellOT instructions, or add your local CellOT clone to `PYTHONPATH`:

```bash
# Example only; adjust to your local setup.
export PYTHONPATH=/path/to/cellot:$PYTHONPATH
```

For the 2i/WOT time-course experiments, install the WOT dependency if needed:

```bash
pip install wot
```

---

## Data setup

This repository does **not** redistribute the biological datasets. Please download each dataset from the original study or benchmark website, follow the corresponding license/terms of use, and cite the original data source in any derivative work. The synthetic experiments are generated directly by the notebooks and do not require an external dataset.

| Dataset used in the paper | Original source to use | Notes for this repository |
|---|---|---|
| **AP-1 protein perturbations** | Comandante-Lou, Baumann, and Fallahi-Sichani, *Cell Reports* 2022: [study page / DOI](https://doi.org/10.1016/j.celrep.2022.111147), [PMC version](https://pmc.ncbi.nlm.nih.gov/articles/PMC9395172/), and the authors' [AP1-networkPlasticityMelanoma](https://github.com/fallahi-sichani-lab/AP1-networkPlasticityMelanoma) repository. In the paper, this is the AP-1 benchmark obtained from the original study page and **Supplementary Data S4**. | Used for DMSO $\rightarrow$ VEM protein perturbation prediction across melanoma cell lines. Place locally under `data/ap1/` after downloading. |
| **4i multiplexed protein-imaging perturbations** | Gut et al., *Science* 2018: [paper / DOI](https://doi.org/10.1126/science.aar7042). For the benchmark format used by CellOT, use Bunne et al., *Nature Methods* 2023: [CellOT repository](https://github.com/bunnech/cellot) and [ETH Research Collection processed datasets](https://doi.org/10.3929/ethz-b-000609681). | Used for drug-response prediction after 8h exposure. Place locally under `data/4i/`. |
| **SciPlex3 scRNA-seq perturbations** | Srivatsan et al., *Science* 2020: [paper / DOI](https://doi.org/10.1126/science.aax6234), [NCBI GEO GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944), and the authors' [sci-plex code repository](https://github.com/cole-trapnell-lab/sci-plex). The CellOT-preprocessed version can also be obtained from the CellOT processed datasets above. | Used for 24h transcriptomic drug-response prediction. Place locally under `data/sciplex3/`. |
| **2i reprogramming time-course** | Schiebinger et al., *Cell* 2019: [paper / DOI](https://doi.org/10.1016/j.cell.2019.01.006) and the [Waddington-OT tutorial/data page](https://broadinstitute.github.io/wot/tutorial/), which links the tutorial input data and transport maps. | Used for the 12h--168h time-gap sweep. Place locally under `data/reprogramming_2i/`. |

---

## Evaluation metric

The main evaluation metric is the squared Maximum Mean Discrepancy, MMD$^2$, with a Gaussian RBF kernel:

$$
k_\gamma(u,v) = \exp\left(-\gamma \|u-v\|_2^2\right).
$$

The paper reports MMD$^2$ under median-heuristic bandwidths and fixed-bandwidth sensitivity checks. For high-dimensional scRNA-seq experiments, MMD$^2$ is computed in highly variable gene subspaces to improve stability and comparability.

---

## Reproducibility notes

- Use the same train/test split protocol as the corresponding notebook.
- For MMD$^2$, select the RBF bandwidth from the training split only to avoid test leakage.
- LAOT has effectively deterministic behavior after the split is fixed, because it uses a linear assignment step and a least-squares map update.
- Neural baselines can have nontrivial variance across random seeds. Report mean ± standard deviation over repeated runs.
- The untrackable regime should be interpreted as a computational/statistical barrier, not merely as a failure of one solver.

---

## Repository topics

`single-cell` · `perturbation-prediction` · `optimal-transport` · `linear-assignment` · `cellot` · `scgen` · `mmd` · `icml-2026` · `trackability` · `computational-complexity`

---

## Citation

If this repository is useful for your work, please cite:

```bibtex
@inproceedings{jafari2026temporal,
  title     = {Position: Temporal Measurement Interval Determines Computational and Model Complexity in Single-Cell Perturbation Analysis},
  author    = {Jafari, Alireza and Shakeri, Heman and Daneshmand, Hadi},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  url       = {https://openreview.net/forum?id=lECKpTE1lW}
}
```

---

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.
