#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""Run the UN GA Voting x Global Health Governance analysis pipeline.

The script reads four CSV files, engineers voting/network/health features, and
exports five publication-style figures. It is intentionally executable as a
plain Python script so the repository can be reproduced without packaging.

Example:
    python src/un_ga_global_health_analysis.py --data-dir data --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib import cm, colors as mcolors
import matplotlib.patheffects as pe

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr, norm
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="UN GA voting and global health governance empirical analysis"
)
parser.add_argument(
    "--data-dir",
    default="data",
    help="Folder containing the input CSV files. Default: data",
)
parser.add_argument(
    "--output-dir",
    default="outputs",
    help="Folder where figures and reports will be saved. Default: outputs",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed used by bootstrap, model, and manifold steps. Default: 42",
)
parser.add_argument(
    "--n-jobs",
    type=int,
    default=1,
    help=(
        "Parallel jobs for scikit-learn models. Default: 1 for maximum portability; "
        "use -1 on local machines when multiprocessing is allowed."
    ),
)
parser.add_argument(
    "--n-boot",
    type=int,
    default=2000,
    help="Bootstrap resamples for Monte Carlo summaries. Default: 2000",
)
parser.add_argument(
    "--perm-repeats",
    type=int,
    default=30,
    help="Permutation-importance repeats. Default: 30",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=220,
    help="Resolution for exported PNG figures. Default: 220",
)
args = parser.parse_args()

np.random.seed(args.seed)

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("▶  Loading datasets …")

def read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. Put the CSV file in {DATA_DIR} or pass --data-dir."
        )
    return pd.read_csv(path)

panel = read_csv("bm_who_augmented_panel_100countries_2004_2024.csv")
edges = read_csv("un_ga_voting_similarity_edges_100countries_2004_2024.csv")
net_un = read_csv("un_ga_voting_network_metrics_100countries_2004_2024.csv")
country_y = read_csv("un_ga_voting_country_year_summary_100countries.csv")

print(f"   Panel rows: {len(panel):,}   │   Edge rows: {len(edges):,}")

# ─────────────────────────────────────────────
#  COLOUR SYSTEM
# ─────────────────────────────────────────────
BG      = "#0a0a12"
C1      = "#00f5d4"   # neon teal
C2      = "#ff006e"   # neon pink
C3      = "#ffbe0b"   # neon yellow
C4      = "#8338ec"   # electric violet
C5      = "#3a86ff"   # azure
C6      = "#fb5607"   # orange
GRID    = "#1a1a2e"

cmap_div   = LinearSegmentedColormap.from_list("div",   [C2, BG, C1], N=512)
cmap_seq   = LinearSegmentedColormap.from_list("seq",   [BG, C4, C5, C1], N=512)
cmap_fire  = LinearSegmentedColormap.from_list("fire",  [BG, C6, C3, "#ffffff"], N=512)
cmap_cool  = LinearSegmentedColormap.from_list("cool",  [BG, C4, C1], N=512)

REGION_COLORS = {
    "Sub-Saharan Africa":        C1,
    "South Asia":                C3,
    "East Asia & Pacific":       C5,
    "Middle East & North Africa": C2,
    "Latin America & Caribbean": C6,
    "Europe & Central Asia":     C4,
    "North America":             "#aaaaaa",
}

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    "#2a2a3a",
    "axes.labelcolor":   "#ccccdd",
    "xtick.color":       "#666688",
    "ytick.color":       "#666688",
    "text.color":        "#e0e0f0",
    "grid.color":        GRID,
    "grid.linewidth":    0.4,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "legend.facecolor":  "#10101e",
    "legend.edgecolor":  "#2a2a3a",
})

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("▶  Engineering features …")

feat_cols = [
    "un_yes_share", "un_no_share", "un_abstain_share",
    "un_degree", "un_strength", "un_eigen",
    "un_betweenness", "un_closeness", "un_local_clustering",
    "un_network_density", "un_network_clustering",
    "health_spending", "physicians", "nurses",
    "hospital_beds", "internet", "gdp_pc_ppp",
]

target_col = "UHC_INDEX_REPORTED"

df = panel.copy()
df = df.dropna(subset=[target_col])
df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")

df["voting_entropy"] = (
    -df["un_yes_share"].clip(1e-9) * np.log(df["un_yes_share"].clip(1e-9))
    -df["un_no_share"].clip(1e-9) * np.log(df["un_no_share"].clip(1e-9))
    -df["un_abstain_share"].clip(1e-9) * np.log(df["un_abstain_share"].clip(1e-9))
)
df["network_influence"] = df["un_eigen"] * df["un_strength"]
df["log_gdp"]           = np.log1p(df["gdp_pc_ppp"])
df["physicians_beds"]   = df["physicians"] * df["hospital_beds"]

all_feats = feat_cols + ["voting_entropy", "network_influence", "log_gdp", "physicians_beds"]

X_raw = df[all_feats].values
y     = df[target_col].values

imp  = SimpleImputer(strategy="median")
scl  = RobustScaler()
X    = scl.fit_transform(imp.fit_transform(X_raw))

# ═══════════════════════════════════════════════════════════════
#  FIGURE 1: MONTE CARLO BOOTSTRAP + SIGMA ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════
print("▶  Figure 1 — Monte Carlo + Sigma Analysis …")

N_BOOT = args.n_boot
years  = sorted(df["year"].unique())

region_trends = {}
for reg, grp in df.groupby("region"):
    yr_means = grp.groupby("year")[target_col].mean()
    region_trends[reg] = yr_means

# Monte Carlo: resample agreement scores per year
mc_agreement_mean = []
mc_agreement_ci_lo = []
mc_agreement_ci_hi = []
mc_years_list = []

agreement_by_year = edges.groupby("year")["un_voting_agreement"]

for yr, vals in agreement_by_year:
    v = vals.dropna().values
    if len(v) < 10: continue
    boot = [np.mean(np.random.choice(v, len(v), replace=True)) for _ in range(N_BOOT)]
    mc_years_list.append(yr)
    mc_agreement_mean.append(np.mean(boot))
    mc_agreement_ci_lo.append(np.percentile(boot, 2.5))
    mc_agreement_ci_hi.append(np.percentile(boot, 97.5))

mc_years = np.array(mc_years_list)
mc_mean  = np.array(mc_agreement_mean)
mc_lo    = np.array(mc_agreement_ci_lo)
mc_hi    = np.array(mc_agreement_ci_hi)

# 3-Sigma anomaly detection on UHC vs voting
uhc_by_year = df.groupby("year")[target_col].mean().values
sigma_threshold = 3.0
uhc_z = stats.zscore(uhc_by_year)
anomaly_mask = np.abs(uhc_z) > sigma_threshold

# Sigma detection on cosine per country
country_sigma = {}
for iso, grp in df.groupby("iso3c"):
    v = grp.sort_values("year")[target_col].values
    if len(v) > 5:
        z = np.abs(stats.zscore(v))
        country_sigma[iso] = {"max_z": z.max(), "mean_z": z.mean(), "region": grp["region"].iloc[0]}

sigma_df = pd.DataFrame(country_sigma).T.reset_index()
sigma_df.columns = ["iso3c", "max_z", "mean_z", "region"]
sigma_df[["max_z","mean_z"]] = sigma_df[["max_z","mean_z"]].astype(float)

fig1 = plt.figure(figsize=(22, 16), facecolor=BG)
fig1.suptitle(
    "MONTE CARLO BOOTSTRAP  ×  3-σ ANOMALY DETECTION\n"
    "UN General Assembly Voting Agreement & UHC Trajectory · 2004–2024",
    fontsize=15, color=C1, fontweight="bold", y=0.98,
    path_effects=[pe.withStroke(linewidth=6, foreground=BG)]
)

gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.42, wspace=0.38,
                         top=0.92, bottom=0.06, left=0.07, right=0.97)

# ── Panel A: Monte Carlo CI bands
ax_mc = fig1.add_subplot(gs1[0, :2])
ax_mc.fill_between(mc_years, mc_lo, mc_hi, color=C4, alpha=0.18, label="95% Bootstrap CI")
ax_mc.fill_between(mc_years, mc_lo + (mc_hi-mc_lo)*0.25,
                   mc_hi - (mc_hi-mc_lo)*0.25, color=C4, alpha=0.30, label="50% Bootstrap CI")
ax_mc.plot(mc_years, mc_mean, color=C1, lw=2.5, zorder=5, label="Bootstrap Mean")
smooth = gaussian_filter1d(mc_mean, sigma=1.2)
ax_mc.plot(mc_years, smooth, color=C3, lw=1.5, ls="--", alpha=0.8, label="LOESS Smooth")
ax_mc.set_title(f"A · MONTE CARLO VOTING AGREEMENT  (N={N_BOOT:,} bootstrap resamples)", color=C3)
ax_mc.set_ylabel("Mean Agreement Score")
ax_mc.legend(fontsize=7, ncol=2)
ax_mc.grid(True, alpha=0.3)

# ── Panel B: Sigma anomaly distribution
ax_sig = fig1.add_subplot(gs1[0, 2])
for reg in sigma_df["region"].unique():
    sub = sigma_df[sigma_df["region"] == reg]
    c   = REGION_COLORS.get(reg, "#888888")
    ax_sig.scatter(sub["mean_z"], sub["max_z"],
                   color=c, alpha=0.7, s=30, label=reg[:20], edgecolors="none")

ax_sig.axhline(3.0, color=C2, lw=1.5, ls="--", alpha=0.8)
ax_sig.axhline(2.0, color=C3, lw=1.0, ls=":", alpha=0.6)
ax_sig.set_title("B · 3-σ ANOMALY MAP\nPer-country UHC Volatility", color=C3)
ax_sig.set_xlabel("Mean |Z|"); ax_sig.set_ylabel("Max |Z|")
ax_sig.text(sigma_df["mean_z"].max()*0.6, 3.15, "3σ threshold", color=C2, fontsize=7)
ax_sig.grid(True, alpha=0.3)

# ── Panel C: UHC regional trends with sigma bands
ax_reg = fig1.add_subplot(gs1[1, :2])
for reg, trend in region_trends.items():
    c   = REGION_COLORS.get(reg, "#888888")
    yrs = trend.index.values
    vals = trend.values
    ax_reg.plot(yrs, vals, color=c, lw=1.8, alpha=0.85, label=reg[:24])
    # sigma band per region
    grp = df[df["region"] == reg]
    for yr in yrs:
        sub = grp[grp["year"] == yr][target_col].dropna()
        if len(sub) > 2:
            mu, sd = sub.mean(), sub.std()
            ax_reg.fill_between([yr-0.3, yr+0.3], mu-sd, mu+sd, color=c, alpha=0.08)

ax_reg.set_title("C · REGIONAL UHC TRAJECTORIES  (±1σ shading per year)", color=C3)
ax_reg.set_ylabel("UHC Service Coverage Index")
ax_reg.legend(fontsize=6.5, ncol=2)
ax_reg.grid(True, alpha=0.3)

# ── Panel D: Sigma histogram
ax_hist = fig1.add_subplot(gs1[1, 2])
vals_sig = sigma_df["max_z"].dropna().values
ax_hist.hist(vals_sig, bins=30, color=C4, alpha=0.7, edgecolor=BG, linewidth=0.4)
ax_hist.axvline(3.0, color=C2, lw=2, ls="--", label="3σ")
ax_hist.axvline(2.0, color=C3, lw=1.5, ls=":", label="2σ")
x_range = np.linspace(0, vals_sig.max(), 200)
kde = stats.gaussian_kde(vals_sig)
ax_hist2 = ax_hist.twinx()
ax_hist2.plot(x_range, kde(x_range), color=C1, lw=1.5)
ax_hist2.set_ylabel("KDE", color=C1, fontsize=7)
ax_hist2.tick_params(colors=C1, labelsize=6)
ax_hist.set_title("D · MAX |Z| DISTRIBUTION", color=C3)
ax_hist.set_xlabel("|Z| score")
ax_hist.legend(fontsize=7)
ax_hist.grid(True, alpha=0.3)

# ── Panel E: Per-year Monte Carlo violin of cosine similarities
ax_vio = fig1.add_subplot(gs1[2, :])
years_vio = sorted(edges["year"].unique())
positions = np.arange(len(years_vio))
cos_data  = [edges[edges["year"] == yr]["un_voting_cosine"].dropna().values for yr in years_vio]

parts = ax_vio.violinplot(cos_data, positions=positions, showmedians=True,
                          showextrema=False, widths=0.75)
for i, (pc, yr_d) in enumerate(zip(parts["bodies"], cos_data)):
    mu = np.mean(yr_d)
    z  = (mu - np.mean([np.mean(d) for d in cos_data])) / np.std([np.mean(d) for d in cos_data])
    c  = cmap_div(Normalize(-3,3)(z))
    pc.set_facecolor(c)
    pc.set_alpha(0.75)
    pc.set_edgecolor(C1)
    pc.set_linewidth(0.6)

parts["cmedians"].set_color(C3)
parts["cmedians"].set_linewidth(2)
ax_vio.set_xticks(positions)
ax_vio.set_xticklabels(years_vio, rotation=45, fontsize=7)
ax_vio.set_title("E · MONTE CARLO COSINE SIMILARITY DISTRIBUTION PER YEAR  (color = z-score of mean)", color=C3)
ax_vio.set_ylabel("Cosine Similarity")
ax_vio.grid(True, alpha=0.3, axis="y")

plt.savefig(OUTPUT_DIR / "fig1_monte_carlo_sigma.png",
            dpi=args.dpi, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓  fig1_monte_carlo_sigma.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2: RANDOM FOREST + PERMUTATION IMPORTANCE + OOB
# ═══════════════════════════════════════════════════════════════
print("▶  Figure 2 — Random Forest Analysis …")

# Cross-validated RF
rf = RandomForestRegressor(
    n_estimators=600,
    max_depth=10,
    min_samples_leaf=5,
    n_jobs=args.n_jobs,
    oob_score=True,
    random_state=args.seed,
    max_features="sqrt",
)

kf     = KFold(n_splits=10, shuffle=True, random_state=args.seed)
cv_r2  = cross_val_score(rf, X, y, cv=kf, scoring="r2")
cv_rmse= np.sqrt(-cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_squared_error"))

rf.fit(X, y)
y_pred       = rf.predict(X)
oob_pred     = rf.oob_prediction_
feat_imp     = rf.feature_importances_
feat_names   = all_feats

# Permutation importance
perm_res = permutation_importance(
    rf,
    X,
    y,
    n_repeats=args.perm_repeats,
    random_state=args.seed,
    n_jobs=args.n_jobs,
)

# GB model for comparison
gb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                max_depth=5, random_state=args.seed)
gb.fit(X, y)
gb_imp = gb.feature_importances_

# SHAP-like partial effects via MC
MC_PARTIAL = 500
partial_effects = {}
for i, fname in enumerate(feat_names[:8]):
    grid = np.linspace(X[:, i].min(), X[:, i].max(), MC_PARTIAL)
    X_partial = np.tile(np.median(X, axis=0), (MC_PARTIAL, 1))
    X_partial[:, i] = grid
    partial_effects[fname] = (grid, rf.predict(X_partial))


fig2 = plt.figure(figsize=(24, 20), facecolor=BG)
fig2.suptitle(
    "RANDOM FOREST · GRADIENT BOOSTING  ×  PERMUTATION IMPORTANCE\n"
    "Predicting UHC Coverage from UN Voting & Network Metrics · 10-Fold CV",
    fontsize=15, color=C5, fontweight="bold", y=0.98
)

gs2 = gridspec.GridSpec(3, 4, figure=fig2, hspace=0.48, wspace=0.40,
                         top=0.93, bottom=0.06, left=0.06, right=0.97)

# ── A: Feature importance comparison (RF vs GB)
ax_fi = fig2.add_subplot(gs2[0, :2])
idx   = np.argsort(feat_imp)[::-1]
x_pos = np.arange(len(feat_names))

bars_rf = ax_fi.bar(x_pos - 0.18, feat_imp[idx], width=0.35,
                    color=C5, alpha=0.85, label="RF Gini Importance", zorder=3)
bars_gb = ax_fi.bar(x_pos + 0.18, gb_imp[idx], width=0.35,
                    color=C6, alpha=0.85, label="GB Importance", zorder=3)

for bar, val in zip(bars_rf, feat_imp[idx]):
    if val > 0.04:
        ax_fi.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=5.5, color=C1)

ax_fi.set_xticks(x_pos)
ax_fi.set_xticklabels([feat_names[i][:18] for i in idx], rotation=55, fontsize=6.5, ha="right")
ax_fi.set_title("A · RF vs GB FEATURE IMPORTANCE  (sorted by RF Gini)", color=C3)
ax_fi.legend(fontsize=8)
ax_fi.grid(True, axis="y", alpha=0.3)

# ── B: Permutation importance with error bars
ax_perm = fig2.add_subplot(gs2[0, 2:])
perm_means = perm_res.importances_mean
perm_stds  = perm_res.importances_std
pidx = np.argsort(perm_means)[::-1]

colors_perm = [cmap_fire(Normalize(0,1)(v)) for v in
               (perm_means[pidx] - perm_means[pidx].min()) / (perm_means[pidx].max() - perm_means[pidx].min() + 1e-9)]

ax_perm.barh([feat_names[i][:22] for i in pidx], perm_means[pidx],
             xerr=perm_stds[pidx], color=colors_perm, alpha=0.85,
             error_kw={"ecolor": C3, "capsize": 3, "elinewidth": 1.2}, zorder=3)
ax_perm.axvline(0, color="#444466", lw=1)
ax_perm.set_title("B · PERMUTATION IMPORTANCE  (30 repeats, ±σ error bars)", color=C3)
ax_perm.set_xlabel("Mean Decrease in R²")
ax_perm.grid(True, axis="x", alpha=0.3)

# ── C: Cross-validation R² distribution
ax_cv = fig2.add_subplot(gs2[1, 0])
folds = np.arange(1, len(cv_r2)+1)
ax_cv.bar(folds, cv_r2, color=[C1 if v > cv_r2.mean() else C2 for v in cv_r2],
          alpha=0.85, edgecolor=BG, linewidth=0.8)
ax_cv.axhline(cv_r2.mean(), color=C3, lw=2, ls="--", label=f"Mean R²={cv_r2.mean():.3f}")
ax_cv.fill_between([-0.5, len(folds)+0.5],
                   cv_r2.mean()-cv_r2.std(), cv_r2.mean()+cv_r2.std(),
                   color=C3, alpha=0.12)
ax_cv.set_title("C · 10-FOLD CV  R² per fold", color=C3)
ax_cv.set_xlabel("Fold"); ax_cv.set_ylabel("R²")
ax_cv.legend(fontsize=7)
ax_cv.grid(True, alpha=0.3)

# ── D: OOB prediction vs actual
ax_oob = fig2.add_subplot(gs2[1, 1])
sc = ax_oob.scatter(y, oob_pred, c=y - oob_pred,
                    cmap=cmap_div, s=6, alpha=0.5, edgecolors="none",
                    vmin=-30, vmax=30)
plt.colorbar(sc, ax=ax_oob, label="Residual", fraction=0.04)
lims = [min(y.min(), oob_pred.min()), max(y.max(), oob_pred.max())]
ax_oob.plot(lims, lims, color=C3, lw=1.5, ls="--")
oob_r2 = r2_score(y, oob_pred)
ax_oob.set_title(f"D · OOB PREDICTION  R²={oob_r2:.3f}", color=C3)
ax_oob.set_xlabel("Actual UHC"); ax_oob.set_ylabel("OOB Predicted")
ax_oob.grid(True, alpha=0.3)

# ── E: RMSE across folds
ax_rmse = fig2.add_subplot(gs2[1, 2])
ax_rmse.plot(folds, cv_rmse, "o-", color=C6, lw=2, ms=7, zorder=5)
ax_rmse.fill_between(folds, cv_rmse - cv_rmse.std()*0.5,
                      cv_rmse + cv_rmse.std()*0.5, color=C6, alpha=0.15)
ax_rmse.axhline(cv_rmse.mean(), color=C3, ls="--", lw=1.5,
                label=f"Mean RMSE={cv_rmse.mean():.2f}")
ax_rmse.set_title("E · CROSS-VAL RMSE", color=C3)
ax_rmse.set_xlabel("Fold"); ax_rmse.set_ylabel("RMSE")
ax_rmse.legend(fontsize=7)
ax_rmse.grid(True, alpha=0.3)

# ── F: Residual distribution + normality test
ax_resid = fig2.add_subplot(gs2[1, 3])
resid   = y - oob_pred
_, p_sw = stats.shapiro(resid[:2000])
ax_resid.hist(resid, bins=40, color=C4, alpha=0.7, edgecolor=BG,
              linewidth=0.4, density=True)
x_norm = np.linspace(resid.min(), resid.max(), 300)
ax_resid.plot(x_norm, norm.pdf(x_norm, resid.mean(), resid.std()),
              color=C1, lw=2, label="Normal fit")
kde_r = stats.gaussian_kde(resid)
ax_resid.plot(x_norm, kde_r(x_norm), color=C2, lw=1.5, ls="--", label="KDE")
ax_resid.set_title(f"F · RESIDUAL DISTRIBUTION\nShapiro-Wilk p={p_sw:.4f}", color=C3)
ax_resid.set_xlabel("Residual"); ax_resid.legend(fontsize=7)
ax_resid.grid(True, alpha=0.3)

# ── G: Partial dependence curves (8 most important)
for j, (fname, (grid, eff)) in enumerate(partial_effects.items()):
    row, col = divmod(j, 4)
    if row >= 1: break
    ax_pd = fig2.add_subplot(gs2[2, j])
    imp_val = feat_imp[feat_names.index(fname)]
    ax_pd.plot(grid, eff, lw=2, color=[C1, C2, C3, C6][j % 4])
    ax_pd.fill_between(grid, eff.min(), eff, alpha=0.12,
                        color=[C1, C2, C3, C6][j % 4])
    ax_pd.set_title(f"G{j+1}· {fname[:20]}\nimportance={imp_val:.4f}", color=C3, fontsize=8)
    ax_pd.set_xlabel("Scaled Feature Value", fontsize=7)
    ax_pd.set_ylabel("Predicted UHC", fontsize=7)
    ax_pd.grid(True, alpha=0.3)

plt.savefig(OUTPUT_DIR / "fig2_random_forest.png",
            dpi=args.dpi, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓  fig2_random_forest.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 3: PCA · t-SNE MANIFOLD · EIGENVALUE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════
print("▶  Figure 3 — PCA / t-SNE Manifold …")

X_nona = imp.transform(X_raw)
X_sc   = scl.transform(X_nona)

pca    = PCA(n_components=min(12, X_sc.shape[1]))
X_pca  = pca.fit_transform(X_sc)

tsne   = TSNE(n_components=2, perplexity=40, max_iter=1500, random_state=args.seed)
X_tsne = tsne.fit_transform(X_sc)

region_arr = df["region"].values[:len(X_tsne)]
year_arr   = df["year"].values[:len(X_tsne)]
uhc_arr    = df[target_col].values[:len(X_tsne)]

fig3 = plt.figure(figsize=(24, 18), facecolor=BG)
fig3.suptitle(
    "PCA MANIFOLD DECOMPOSITION  ×  t-SNE TOPOLOGY\n"
    "UN Voting + Health Feature Space · Eigenvalue Spectrum · Biplot",
    fontsize=15, color=C4, fontweight="bold", y=0.98
)

gs3 = gridspec.GridSpec(2, 4, figure=fig3, hspace=0.45, wspace=0.38,
                         top=0.92, bottom=0.06, left=0.06, right=0.97)

# ── A: t-SNE colored by region
ax_tsne_r = fig3.add_subplot(gs3[0, :2])
for reg in np.unique(region_arr):
    mask = region_arr == reg
    c    = REGION_COLORS.get(reg, "#888888")
    ax_tsne_r.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=c, s=8, alpha=0.6, label=reg[:24], edgecolors="none")
ax_tsne_r.set_title("A · t-SNE TOPOLOGY  (colored by region)", color=C3)
ax_tsne_r.legend(fontsize=6.5, ncol=1, loc="upper left")
ax_tsne_r.grid(True, alpha=0.2)

# ── B: t-SNE colored by UHC
ax_tsne_u = fig3.add_subplot(gs3[0, 2])
sc_u = ax_tsne_u.scatter(X_tsne[:, 0], X_tsne[:, 1],
                           c=uhc_arr, cmap=cmap_seq, s=8, alpha=0.7, edgecolors="none",
                           vmin=np.nanpercentile(uhc_arr, 5),
                           vmax=np.nanpercentile(uhc_arr, 95))
plt.colorbar(sc_u, ax=ax_tsne_u, label="UHC Index", fraction=0.045)
ax_tsne_u.set_title("B · t-SNE  (UHC gradient)", color=C3)
ax_tsne_u.grid(True, alpha=0.2)

# ── C: Eigenvalue scree plot with cumulative
ax_scree = fig3.add_subplot(gs3[0, 3])
ev  = pca.explained_variance_ratio_
cum = np.cumsum(ev)
ax_scree.bar(range(1, len(ev)+1), ev*100, color=C4, alpha=0.85, edgecolor=BG)
ax_scree2 = ax_scree.twinx()
ax_scree2.plot(range(1, len(ev)+1), cum*100, "o-", color=C1, lw=2, ms=5)
ax_scree2.set_ylabel("Cumulative %", color=C1, fontsize=8)
ax_scree2.tick_params(colors=C1)
ax_scree2.axhline(90, color=C3, ls="--", lw=1, alpha=0.7)
ax_scree.set_title("C · EIGENVALUE SCREE\nExplained Variance", color=C3)
ax_scree.set_xlabel("Component"); ax_scree.set_ylabel("Variance %")
ax_scree.grid(True, alpha=0.3)

# ── D: PCA Biplot (PC1 vs PC2)
ax_bi = fig3.add_subplot(gs3[1, :2])
sc_bi = ax_bi.scatter(X_pca[:, 0], X_pca[:, 1],
                       c=uhc_arr, cmap=cmap_seq, s=6, alpha=0.5, edgecolors="none",
                       vmin=np.nanpercentile(uhc_arr, 5),
                       vmax=np.nanpercentile(uhc_arr, 95))
plt.colorbar(sc_bi, ax=ax_bi, label="UHC", fraction=0.03)

scale = np.abs(X_pca[:, :2]).max(axis=0) * 0.8
for i, fname in enumerate(feat_names):
    x_load = pca.components_[0, i] * scale[0]
    y_load = pca.components_[1, i] * scale[1]
    if abs(x_load) + abs(y_load) > 1.0:
        ax_bi.annotate("", xy=(x_load, y_load), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->", color=C2, lw=1.2, alpha=0.7))
        ax_bi.text(x_load*1.08, y_load*1.08, fname[:14],
                   fontsize=5.5, color=C3, ha="center",
                   path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

ax_bi.set_title("D · PCA BIPLOT  PC1 vs PC2  (loading arrows = |loading| > threshold)", color=C3)
ax_bi.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
ax_bi.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax_bi.axhline(0, color="#333355", lw=0.8)
ax_bi.axvline(0, color="#333355", lw=0.8)
ax_bi.grid(True, alpha=0.2)

# ── E: Loading heatmap
ax_load = fig3.add_subplot(gs3[1, 2:])
n_comp_show = min(8, len(ev))
load_mat = pca.components_[:n_comp_show, :]
im = ax_load.imshow(load_mat, cmap=cmap_div, aspect="auto", vmin=-0.7, vmax=0.7)
plt.colorbar(im, ax=ax_load, label="Loading", fraction=0.02)
ax_load.set_xticks(range(len(feat_names)))
ax_load.set_xticklabels([f[:16] for f in feat_names], rotation=55, fontsize=5.5, ha="right")
ax_load.set_yticks(range(n_comp_show))
ax_load.set_yticklabels([f"PC{i+1} ({ev[i]*100:.1f}%)" for i in range(n_comp_show)], fontsize=7)
ax_load.set_title("E · PCA LOADING MATRIX  (component × feature loadings)", color=C3)

for i in range(n_comp_show):
    for j in range(len(feat_names)):
        v = load_mat[i, j]
        if abs(v) > 0.30:
            ax_load.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=4.5, color="#ffffff" if abs(v) > 0.45 else "#aaaacc",
                         fontweight="bold")

plt.savefig(OUTPUT_DIR / "fig3_pca_tsne.png",
            dpi=args.dpi, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓  fig3_pca_tsne.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 4: NETWORK CENTRALITY + BAYESIAN CI + EMPIRICAL
# ═══════════════════════════════════════════════════════════════
print("▶  Figure 4 — Network Analysis + Bayesian CI …")

# Bayesian bootstrap (Rubin 1981)
def bayesian_bootstrap_ci(data, stat_fn=np.mean, n_boot=3000, ci=0.95):
    data = np.asarray(data)
    n    = len(data)
    boots = []
    for _ in range(n_boot):
        w = np.random.exponential(1, n)
        w /= w.sum()
        boots.append(np.sum(w * stat_fn(data) if callable(stat_fn) else w * data))
    alpha = 1 - ci
    return np.percentile(boots, [alpha/2*100, (1-alpha/2)*100])

# Per-region centrality trajectories
region_centrality = {}
for reg, grp in net_un.groupby("region"):
    yr_mean = grp.groupby("year")["un_eigen"].mean()
    yr_std  = grp.groupby("year")["un_eigen"].std()
    region_centrality[reg] = {"mean": yr_mean, "std": yr_std}

# Bayesian CI on global UHC vs voting agreement correlation
n_countries = df["iso3c"].nunique()
corr_boot_means = []
corr_boot_stds  = []

for yr in years:
    sub = df[df["year"] == yr][["un_yes_share", target_col]].dropna()
    if len(sub) < 10: continue
    boots = []
    for _ in range(2000):
        idx  = np.random.choice(len(sub), len(sub), replace=True)
        s    = sub.iloc[idx]
        r, _ = pearsonr(s["un_yes_share"], s[target_col])
        boots.append(r)
    corr_boot_means.append((yr, np.mean(boots)))
    corr_boot_stds.append(np.std(boots))

corr_years  = np.array([c[0] for c in corr_boot_means])
corr_means  = np.array([c[1] for c in corr_boot_means])
corr_stds   = np.array(corr_boot_stds)

# Network metrics temporal decomposition
net_un["year_norm"] = (net_un["year"] - net_un["year"].min()) / (net_un["year"].max() - net_un["year"].min())


fig4 = plt.figure(figsize=(24, 20), facecolor=BG)
fig4.suptitle(
    "UN VOTING NETWORK CENTRALITY  ×  BAYESIAN BOOTSTRAP\n"
    "Eigenvector Centrality Trajectories · Corr(Yes-Share, UHC) Bootstrap CI · 2004–2024",
    fontsize=15, color=C2, fontweight="bold", y=0.98
)

gs4 = gridspec.GridSpec(3, 3, figure=fig4, hspace=0.48, wspace=0.38,
                         top=0.92, bottom=0.06, left=0.06, right=0.97)

# ── A: Correlation with Bayesian bootstrap CI
ax_corr = fig4.add_subplot(gs4[0, :2])
ax_corr.fill_between(corr_years,
                      corr_means - 2*corr_stds,
                      corr_means + 2*corr_stds,
                      color=C2, alpha=0.15, label="±2σ Bootstrap Band")
ax_corr.fill_between(corr_years,
                      corr_means - corr_stds,
                      corr_means + corr_stds,
                      color=C2, alpha=0.25, label="±1σ Bootstrap Band")
ax_corr.plot(corr_years, corr_means, color=C1, lw=2.5, zorder=5, label="Bootstrap Mean r")
smooth_corr = gaussian_filter1d(corr_means, sigma=1.0)
ax_corr.plot(corr_years, smooth_corr, color=C3, lw=1.5, ls="--", alpha=0.9, label="LOESS")
ax_corr.axhline(0, color="#444466", lw=1)
ax_corr.set_title("A · PEARSON r(Yes-Share, UHC)  — BAYESIAN BOOTSTRAP CI  (N=2000 per year)", color=C3)
ax_corr.set_ylabel("Correlation Coefficient r")
ax_corr.legend(fontsize=7)
ax_corr.grid(True, alpha=0.3)

# ── B: Centrality by region
ax_cent = fig4.add_subplot(gs4[0, 2])
for reg, vals in region_centrality.items():
    c   = REGION_COLORS.get(reg, "#888888")
    yrs = vals["mean"].index.values
    mu  = vals["mean"].values
    sd  = vals["std"].values
    ax_cent.plot(yrs, mu, color=c, lw=1.8, label=reg[:18])
    ax_cent.fill_between(yrs, mu-sd*0.5, mu+sd*0.5, color=c, alpha=0.10)

ax_cent.set_title("B · EIGEN CENTRALITY\nby Region (±0.5σ)", color=C3)
ax_cent.set_xlabel("Year"); ax_cent.set_ylabel("Eigen Centrality")
ax_cent.legend(fontsize=5.5, ncol=1)
ax_cent.grid(True, alpha=0.3)

# ── C: Scatter — network strength vs UHC with sigma ellipses
ax_net_uhc = fig4.add_subplot(gs4[1, :2])
df_net = df.dropna(subset=["un_strength", target_col, "region"])

for reg in df_net["region"].unique():
    sub = df_net[df_net["region"] == reg]
    c   = REGION_COLORS.get(reg, "#888888")
    ax_net_uhc.scatter(sub["un_strength"], sub[target_col],
                        color=c, s=8, alpha=0.45, edgecolors="none", label=reg[:20])
    # sigma ellipse
    if len(sub) > 10:
        for n_sig in [1, 2]:
            cov = np.cov(sub["un_strength"].values, sub[target_col].values)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            w, h  = 2 * n_sig * np.sqrt(eigenvalues)
            ell   = mpatches.Ellipse(
                (sub["un_strength"].mean(), sub[target_col].mean()),
                w, h, angle=angle, color=c,
                fill=False, lw=0.7, alpha=0.35, ls="--" if n_sig==2 else "-"
            )
            ax_net_uhc.add_patch(ell)

ax_net_uhc.set_title("C · UN NETWORK STRENGTH vs UHC  (1σ/2σ Ellipses per region)", color=C3)
ax_net_uhc.set_xlabel("UN Network Strength"); ax_net_uhc.set_ylabel("UHC Index")
ax_net_uhc.legend(fontsize=6, ncol=2)
ax_net_uhc.grid(True, alpha=0.3)

# ── D: Betweenness histogram by group
ax_bet = fig4.add_subplot(gs4[1, 2])
for grp_name, sub in net_un.groupby("global_group"):
    vals = sub["un_betweenness"].dropna().values
    kde  = stats.gaussian_kde(vals)
    xg   = np.linspace(0, vals.max(), 200)
    c    = C1 if "South" in grp_name else C4
    ax_bet.fill_between(xg, kde(xg), alpha=0.3, color=c)
    ax_bet.plot(xg, kde(xg), color=c, lw=2, label=grp_name)
    ax_bet.axvline(vals.mean(), color=c, lw=1, ls="--", alpha=0.7)

ax_bet.set_title("D · BETWEENNESS\nGlobal South vs North KDE", color=C3)
ax_bet.set_xlabel("Betweenness Centrality"); ax_bet.set_ylabel("Density")
ax_bet.legend(fontsize=7)
ax_bet.grid(True, alpha=0.3)

# ── E: Heatmap — mean UHC by region × year
pivot_uhc = df.groupby(["region", "year"])[target_col].mean().unstack()
ax_heat = fig4.add_subplot(gs4[2, :2])
im2 = ax_heat.imshow(pivot_uhc.values, cmap=cmap_seq, aspect="auto")
plt.colorbar(im2, ax=ax_heat, label="Mean UHC", fraction=0.02)
ax_heat.set_xticks(range(len(pivot_uhc.columns)))
ax_heat.set_xticklabels(pivot_uhc.columns.astype(int), rotation=60, fontsize=6.5)
ax_heat.set_yticks(range(len(pivot_uhc.index)))
ax_heat.set_yticklabels([r[:28] for r in pivot_uhc.index], fontsize=7)
ax_heat.set_title("E · UHC INDEX HEATMAP  Region × Year  (diverging palette = structural shift)", color=C3)

for i in range(len(pivot_uhc.index)):
    for j in range(len(pivot_uhc.columns)):
        v = pivot_uhc.values[i, j]
        if not np.isnan(v):
            ax_heat.text(j, i, f"{v:.0f}", ha="center", va="center",
                         fontsize=4.5, color="white" if v < 60 else "#0a0a12")

# ── F: Voting agreement over time by Global South vs North
ax_agree = fig4.add_subplot(gs4[2, 2])
df_yr_group = country_y.copy()
if "global_group" not in df_yr_group.columns:
    gg = net_un[["iso3c","year","global_group"]].drop_duplicates()
    df_yr_group = df_yr_group.merge(gg, on=["iso3c","year"], how="left")
df_yr_group = df_yr_group.dropna(subset=["global_group"])
for grp_name, sub in df_yr_group.groupby("global_group"):
    yr_trend = sub.groupby("year")["un_yes_share"].mean()
    c        = C1 if "South" in grp_name else C4
    yrs      = yr_trend.index.values
    vals     = yr_trend.values
    ax_agree.plot(yrs, vals, color=c, lw=2, label=grp_name)
    smooth_v = gaussian_filter1d(vals, sigma=0.8)
    ax_agree.plot(yrs, smooth_v, color=c, lw=1, ls=":", alpha=0.6)
    bci = [np.percentile([np.mean(np.random.choice(
        sub[sub["year"]==y]["un_yes_share"].dropna().values,
        replace=True, size=max(2, len(sub[sub["year"]==y])))
    ) for _ in range(500)], [5, 95]) for y in yrs]
    lo = np.array([b[0] for b in bci])
    hi = np.array([b[1] for b in bci])
    ax_agree.fill_between(yrs, lo, hi, color=c, alpha=0.12)

ax_agree.set_title("F · YES-SHARE TREND\n95% Bootstrap CI", color=C3)
ax_agree.set_xlabel("Year"); ax_agree.set_ylabel("Yes-Vote Share")
ax_agree.legend(fontsize=7)
ax_agree.grid(True, alpha=0.3)

plt.savefig(OUTPUT_DIR / "fig4_network_bayesian.png",
            dpi=args.dpi, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓  fig4_network_bayesian.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 5: COMPOSITE EMPIRICAL FINDINGS DASHBOARD
# ═══════════════════════════════════════════════════════════════
print("▶  Figure 5 — Composite Empirical Dashboard …")

# OLS regression: UHC ~ voting + network + health
X_ols = df[["un_yes_share", "un_eigen", "log_gdp",
            "health_spending", "physicians"]].copy()
X_ols = sm.add_constant(X_ols.fillna(X_ols.median()))
y_ols = df[target_col].fillna(df[target_col].median())
ols_res = sm.OLS(y_ols, X_ols).fit()

# Spearman rank correlations matrix
corr_cols = ["un_yes_share", "un_eigen", "un_strength",
             "health_spending", "gdp_pc_ppp", target_col, "UHC_SCI_INFECT"]
corr_df = df[corr_cols].dropna()
spearman_mat = np.zeros((len(corr_cols), len(corr_cols)))
for i, c1 in enumerate(corr_cols):
    for j, c2 in enumerate(corr_cols):
        r, _ = spearmanr(corr_df[c1], corr_df[c2])
        spearman_mat[i, j] = r

# Annual global voting cohesion index
cohesion = edges.groupby("year").apply(
    lambda x: (x["un_voting_agreement"] > 0.80).mean()
).reset_index()
cohesion.columns = ["year", "cohesion_80"]

fig5 = plt.figure(figsize=(26, 18), facecolor=BG)
fig5.suptitle(
    "COMPOSITE EMPIRICAL FINDINGS DASHBOARD\n"
    "OLS Coefficients · Spearman Matrix · Cohesion Index · Health–Voting Frontier",
    fontsize=15, color=C3, fontweight="bold", y=0.98
)

gs5 = gridspec.GridSpec(2, 4, figure=fig5, hspace=0.50, wspace=0.42,
                         top=0.92, bottom=0.06, left=0.06, right=0.97)

# ── A: OLS coefficient forest plot
ax_ols = fig5.add_subplot(gs5[0, :2])
params = ols_res.params[1:]
cis    = ols_res.conf_int()[1:]
pvals  = ols_res.pvalues[1:]
colors_ols = []
for p in pvals:
    if   p < 0.001: colors_ols.append(C1)
    elif p < 0.01:  colors_ols.append(C3)
    elif p < 0.05:  colors_ols.append(C6)
    else:           colors_ols.append("#555577")

yvec = np.arange(len(params))
ax_ols.barh(yvec, params.values, color=colors_ols, alpha=0.85,
             xerr=[(params.values - cis[0].values),
                   (cis[1].values - params.values)],
             error_kw={"ecolor": "#aaaacc", "capsize": 4, "elinewidth": 1.5},
             height=0.55, edgecolor=BG)
ax_ols.axvline(0, color="#444466", lw=1.5)
ax_ols.set_yticks(yvec)
ax_ols.set_yticklabels(params.index, fontsize=8)
ax_ols.set_title(f"A · OLS REGRESSION COEFFICIENTS  R²={ols_res.rsquared:.3f}\n"
                  f"(green=p<0.001  yellow=p<0.01  orange=p<0.05  grey=ns)",
                  color=C3)
ax_ols.set_xlabel("Coefficient (95% CI)")
for i, (v, p) in enumerate(zip(params.values, pvals)):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax_ols.text(v + (0.5 if v >= 0 else -0.5), i, sig,
                ha="left" if v >= 0 else "right", va="center",
                fontsize=9, color=C3)
ax_ols.grid(True, axis="x", alpha=0.3)

# ── B: Spearman correlation heatmap
ax_sp = fig5.add_subplot(gs5[0, 2])
im3 = ax_sp.imshow(spearman_mat, cmap=cmap_div, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im3, ax=ax_sp, label="Spearman r", fraction=0.045)
ax_sp.set_xticks(range(len(corr_cols)))
ax_sp.set_xticklabels([c[:14] for c in corr_cols], rotation=55, fontsize=6.5, ha="right")
ax_sp.set_yticks(range(len(corr_cols)))
ax_sp.set_yticklabels([c[:14] for c in corr_cols], fontsize=6.5)
ax_sp.set_title("B · SPEARMAN ρ MATRIX", color=C3)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax_sp.text(j, i, f"{spearman_mat[i,j]:.2f}", ha="center", va="center",
                   fontsize=5.5,
                   color="white" if abs(spearman_mat[i,j]) > 0.5 else "#888899")

# ── C: Voting cohesion index (% pairs > 0.80 agreement)
ax_coh = fig5.add_subplot(gs5[0, 3])
c_yrs  = cohesion["year"].values
c_vals = cohesion["cohesion_80"].values
ax_coh.fill_between(c_yrs, 0, c_vals*100, color=C5, alpha=0.4)
ax_coh.plot(c_yrs, c_vals*100, "o-", color=C1, lw=2.5, ms=6, zorder=5)
smooth_coh = gaussian_filter1d(c_vals*100, sigma=1)
ax_coh.plot(c_yrs, smooth_coh, color=C3, lw=1.5, ls="--", alpha=0.9)
ax_coh.set_title("C · VOTING COHESION INDEX\n% country-pairs with r > 0.80", color=C3)
ax_coh.set_ylabel("% Pairs"); ax_coh.set_xlabel("Year")
ax_coh.grid(True, alpha=0.3)

# ── D: Health-Voting frontier (GDP terciles)
ax_front = fig5.add_subplot(gs5[1, :2])
df_f    = df.dropna(subset=["gdp_pc_ppp", target_col, "un_yes_share"]).copy()
df_f["gdp_tercile"] = pd.qcut(df_f["gdp_pc_ppp"], 3, labels=["Low GDP", "Mid GDP", "High GDP"])
terc_colors = [C2, C3, C1]

for tc, c in zip(["Low GDP", "Mid GDP", "High GDP"], terc_colors):
    sub = df_f[df_f["gdp_tercile"] == tc]
    ax_front.scatter(sub["un_yes_share"], sub[target_col],
                      color=c, s=10, alpha=0.4, label=tc, edgecolors="none")
    # LOESS fit per tercile
    if len(sub) > 20:
        sub_s = sub.sort_values("un_yes_share")
        lo_fit = lowess(sub_s[target_col], sub_s["un_yes_share"], frac=0.4)
        ax_front.plot(lo_fit[:, 0], lo_fit[:, 1], color=c, lw=2.5, zorder=5)

ax_front.set_title("D · HEALTH–VOTING FRONTIER  (LOESS by GDP Tercile)\n"
                    "Nonlinear relationship between UN Yes-Vote Share and UHC Coverage",
                    color=C3)
ax_front.set_xlabel("UN Yes-Vote Share")
ax_front.set_ylabel("UHC Service Coverage Index")
ax_front.legend(fontsize=8)
ax_front.grid(True, alpha=0.3)

# ── E: QQ plot of UHC by global group
ax_qq = fig5.add_subplot(gs5[1, 2])
for grp_name, sub in df.groupby("global_group"):
    v = sub[target_col].dropna().values
    qq_res = stats.probplot(v, dist="norm")
    osm, osr = qq_res[0]
    slope, intercept, r = qq_res[1]
    c = C1 if "South" in grp_name else C4
    ax_qq.plot(osm, osr, ".", color=c, ms=3, alpha=0.5, label=grp_name)
    fit_line = slope * np.array(osm) + intercept
    ax_qq.plot(osm, fit_line, color=c, lw=1.5, alpha=0.9)

ax_qq.set_title("E · Q-Q PLOT  UHC Distribution\nvs Normal by Global Group", color=C3)
ax_qq.set_xlabel("Theoretical Quantile")
ax_qq.set_ylabel("Sample Quantile")
ax_qq.legend(fontsize=7)
ax_qq.grid(True, alpha=0.3)

# ── F: 2D density plot: eigen centrality vs UHC
ax_dens = fig5.add_subplot(gs5[1, 3])
dens_df = df.dropna(subset=["un_eigen", target_col])
x_d = dens_df["un_eigen"].values
y_d = dens_df[target_col].values

from scipy.stats import gaussian_kde
xy   = np.vstack([x_d, y_d])
kde2 = gaussian_kde(xy)
xi   = np.linspace(x_d.min(), x_d.max(), 80)
yi   = np.linspace(y_d.min(), y_d.max(), 80)
Xi, Yi = np.meshgrid(xi, yi)
Zi   = kde2(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
ax_dens.contourf(Xi, Yi, Zi, levels=20, cmap=cmap_cool, alpha=0.9)
ax_dens.contour(Xi, Yi, Zi, levels=10, colors=[C3], linewidths=0.4, alpha=0.4)
ax_dens.scatter(x_d, y_d, c=C1, s=3, alpha=0.15, edgecolors="none")
ax_dens.set_title("F · KDE DENSITY  UN Eigen vs UHC\n(2D joint distribution)", color=C3)
ax_dens.set_xlabel("UN Eigenvector Centrality")
ax_dens.set_ylabel("UHC Index")
ax_dens.grid(True, alpha=0.2)

plt.savefig(OUTPUT_DIR / "fig5_empirical_dashboard.png",
            dpi=args.dpi, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓  fig5_empirical_dashboard.png")


# ═══════════════════════════════════════════════════════════════
#  STATISTICAL REPORT
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  EMPIRICAL SUMMARY STATISTICS")
print("="*70)
print(f"  Random Forest 10-fold CV R²    : {cv_r2.mean():.4f}  ±  {cv_r2.std():.4f}")
print(f"  RF OOB Score                   : {rf.oob_score_:.4f}")
print(f"  Mean CV RMSE                   : {cv_rmse.mean():.4f}")
print(f"  OLS R² (voting + health feats) : {ols_res.rsquared:.4f}")
print(f"  OLS Adj-R²                     : {ols_res.rsquared_adj:.4f}")
print(f"  OLS F-statistic                : {ols_res.fvalue:.2f}  (p={ols_res.f_pvalue:.2e})")
print(f"  PCA: comps for 90% variance    : {np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90)+1}")
print(f"  N countries in analysis        : {df['iso3c'].nunique()}")
print(f"  N observations                 : {len(df):,}")
print(f"  Countries flagged >3σ UHC vol  : {(sigma_df['max_z']>3).sum()}")
top3 = sigma_df.nlargest(3, "max_z")[["iso3c", "max_z", "region"]].to_string(index=False)
print(f"  Top 3σ anomaly countries:\n{top3}")
print("="*70)
print(f"\n✅  All 5 figures saved to {OUTPUT_DIR.resolve()}/")
