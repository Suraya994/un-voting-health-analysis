# UN GA Voting x Global Health Governance

Reproducible Python workflow for analysing the relationship between United
Nations General Assembly voting behaviour, network position, and global health
governance indicators.

The repository is designed as a research-oriented empirical analysis pipeline.
It generates publication-style figures using Monte Carlo bootstrap estimation,
random forest modelling, permutation importance, PCA/t-SNE manifold exploration,
network centrality analysis, bootstrap confidence intervals, and regression
diagnostics.

## Repository Structure

```text
.
├── src/
│   ├── un_ga_global_health_analysis.py
│   └── generate_synthetic_data.py
├── data/
├── outputs/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Input Data

The analysis script expects these four CSV files in `data/`:

```text
bm_who_augmented_panel_100countries_2004_2024.csv
un_ga_voting_similarity_edges_100countries_2004_2024.csv
un_ga_voting_network_metrics_100countries_2004_2024.csv
un_ga_voting_country_year_summary_100countries.csv
```

Real research data are not included unless they are public, cleaned, and legally
shareable. The synthetic generator creates schema-compatible artificial data so
reviewers can run the pipeline end to end. Synthetic output is for software
testing only and must not be interpreted as empirical evidence.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Reproducible Demo Run

```bash
python src/generate_synthetic_data.py --output-dir data --seed 42
python src/un_ga_global_health_analysis.py \
  --data-dir data \
  --output-dir outputs \
  --seed 42 \
  --n-jobs 1
```

`--n-jobs 1` is the portable default. On a local machine that supports
multiprocessing, `--n-jobs -1` can speed up model fitting and permutation
importance.

## Main Outputs

The analysis exports five PNG figures:

```text
fig1_monte_carlo_sigma.png
fig2_random_forest.png
fig3_pca_tsne.png
fig4_network_bayesian.png
fig5_empirical_dashboard.png
```

## Notes On Reproducibility

- Random seeds are exposed through `--seed`.
- Input and output folders are configurable through command-line arguments.
- Real CSV files are ignored by git to reduce the risk of accidental data
  disclosure.
- The synthetic demo currently covers the built-in ISO3 sample used by the
  generator while preserving the filenames expected by the empirical pipeline.

## Author

Suraya Bazarova
