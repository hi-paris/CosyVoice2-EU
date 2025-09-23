#!/usr/bin/env python3
"""
Story-first Report Generator for CosyVoice2 (v4.2)

Changes vs v4.1:
  • Cross-language RQ1: fixed crash (“truth value of a Series is ambiguous”).
    Now we check for a dict with 'table_tex' specifically in the caller.

  • Heatmap: switched from Greys to a perceptual color map (default: 'magma_r').

  • Baseline comparison table (B1): row-wise layout.
      For each baseline (including "CosyVoice2 original" = hifigan_only):
        - Row 1: Baseline @ its own best hour
        - Row 2: Ours (best model @ its own best hour) with deltas in parentheses
                 (Δ = ours − baseline; so negative is better for error-type metrics)
      Still writes a CSV (with absolute values & %Δ for provenance).

  • Extra metrics included where available: RTF, F0 RMSE (Hz), F0 Corr, V/UV.
      - In ablations: still keep core columns to avoid over-wide LaTeX.
      - In baseline row-wise table: include extra metrics if present.
      - In cross-language table: keep the 4 core metrics to preserve width.

  • CLI simplified: only --supplemental (enables cross-language, appendix CSVs,
    efficiency, component curves/heatmaps, and mix-vs-mono win/loss).
"""

from __future__ import annotations
import argparse, re, warnings, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------
# Scientific style (Times/IEEE-ish)
# -----------------------------

def set_matplotlib_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.0,
    })

SINGLE_COL_W = 3.39  # inches (IEEE single column)
DOUBLE_COL_W = 7.16  # inches (IEEE double column)

HEATMAP_CMAP = "magma_r"  # perceptual, reversed so lower (better) is brighter

def _finalize_axes(ax, title=None, xlabel=None, ylabel=None, legend=False):
    if title:
        ax.set_title(title, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    if legend:
        leg = ax.legend(frameon=False, handlelength=1.5, borderpad=0.2, loc="best")
        # Matplotlib 3.7+ deprecation: use 'legend_handles'
        for lh in getattr(leg, "legend_handles", []):
            try:
                lh.set_linewidth(1.0)
            except Exception:
                pass

def _save_fig_both(fig, base_no_ext: Path):
    base_no_ext.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = base_no_ext.with_suffix(".pdf")
    png_path = base_no_ext.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return pdf_path, png_path

# -----------------------------
# Config & constants
# -----------------------------

FILENAME_TRAIN_SETTING_RE = re.compile(r"metrics_model-(?P<setting>mix|fr|de)", re.I)

PRIMARY_METRIC_PREF = ["wer_norm", "wer"]
ALL_KNOWN_METRICS = ["wer_norm","wer","cer_norm","cer","secs","mcd","gpe","f0_rmse_hz","f0_corr","vuv","rtf"]

CORE_REPORT_METRICS = ["wer","secs","mcd","f0_corr","vuv"] # "gpe", "rtf"
EXTENDED_REPORT_METRICS = ["rtf","f0_rmse_hz","f0_corr","vuv"]  # included when present
DECIMALS = {
    "mcd":2, "wer":2, "wer_norm":2, "secs":2, "gpe":2, "rtf":2,
    "f0_rmse_hz":2, "f0_corr":2, "vuv":2
}

# Component model sets
# Main set (used for RQ2/RQ3/B1/etc.): keep only variants with a finetuned HiFi-GAN
MAIN_COMPONENT_MODELS = [
    "hifigan_only",   # approximates original CosyVoice2 baseline (no finetuning)
    "llm_hifigan",
    "flow_hifigan",
    "full_finetuned",
]

# Extended set shown only in RQ1 to document ablations and the baseline provenance
RQ1_EXTRA_COMPONENTS = [
    "llm_only",
    "flow_only",
    "llm_flow",
    "pretrained",     # partially trained HiFi-GAN to highlight vocoder effects
]

# Back-compat default if a single list is needed (preserve older behavior)
DEFAULT_COMPONENT_MODELS = [
    "hifigan_only",
    "llm_hifigan",
    "flow_hifigan",
    "full_finetuned",
]

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# -----------------------------
# Loading & hygiene
# -----------------------------

def _infer_train_setting_from_filename(path: Path) -> Optional[str]:
    m = FILENAME_TRAIN_SETTING_RE.search(path.name)
    if not m:
        return None
    s = m.group("setting").lower()
    return "mix" if s == "mix" else "mono"

def load_all_results(results_dir: str,
                     language: Optional[str] = None,
                     backbone: Optional[str] = None) -> pd.DataFrame:
    p = Path(results_dir)
    pattern = f"*{language.lower()}_metrics*.csv" if language else "*metrics*.csv"
    files = list(p.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSVs in {results_dir} matching {pattern}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__file"] = f.name
            for c in ["model","hours","language","backbone","train_setting"]:
                if c not in df.columns:
                    df[c] = np.nan
            if df["train_setting"].isna().all():
                df["train_setting"] = _infer_train_setting_from_filename(f)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Failed to load {f}: {e}")

    if not dfs:
        raise RuntimeError("No valid CSVs loaded")
    df = pd.concat(dfs, ignore_index=True)

    if "language" in df.columns:
        df["language"] = df["language"].astype(str).str.lower()
    if language:
        df = df[df["language"] == language.lower()]
    if backbone and "backbone" in df.columns:
        df = df[df["backbone"].astype(str).str.lower() == backbone.lower()]

    if "hours" in df.columns:
        df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
    for col in ALL_KNOWN_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["train_setting"] = df["train_setting"].fillna("")
    df.loc[df["train_setting"].str.lower().isin(["mix","bilingual"]), "train_setting"] = "mix"
    df.loc[df["train_setting"].str.lower().isin(["mono","fr","de"]), "train_setting"] = "mono"
    return df

# -----------------------------
# Helpers
# -----------------------------

def _pick_primary_metric(df: pd.DataFrame) -> str:
    for m in PRIMARY_METRIC_PREF:
        if m in df.columns and df[m].notna().any():
            return m
    return "wer"

def _effective_wer_col(df_like: pd.DataFrame) -> str:
    if "wer_norm" in df_like.columns and not df_like.empty:
        try:
            if df_like["wer_norm"].notna().any():
                return "wer_norm"
        except Exception:
            pass
    return "wer"

def _agg_mean_std(df: pd.DataFrame, metric: str) -> Tuple[float,float]:
    if metric not in df.columns: return (np.nan, np.nan)
    vals = df[metric].dropna()
    return (float(vals.mean()), float(vals.std(ddof=1))) if len(vals) > 0 else (np.nan, np.nan)

def _fmt(mean: float, std: float, metric: str) -> str:
    if pd.isna(mean):
        return "N/A"
    d = DECIMALS.get(metric, 3)
    if pd.isna(std) or std == 0:
        return f"{mean:.{d}f}"
    return f"{mean:.{d}f} ± {std:.{d}f}"

def _score_direction(metric: str, val: float) -> float:
    if pd.isna(val): return np.inf
    return -val if metric in ["secs"] else val  # higher better only for SECS

def _select_best_setting_and_hour(df: pd.DataFrame,
                                  model: str,
                                  primary: str) -> Tuple[Optional[str], Optional[int]]:
    sub = df[df["model"] == model]
    if sub.empty: return (None, None)
    g = sub.groupby(["train_setting","hours"])[primary].mean().reset_index()
    if g.empty: return (None, None)
    g["_key"] = g[primary].apply(lambda v: _score_direction(primary, v))
    g = g.sort_values("_key", kind="mergesort")
    best = g.iloc[0]
    setting = str(best["train_setting"]) if str(best["train_setting"]) else "mono"
    return (setting, int(best["hours"]))

def _choose_best_model_at_hour(df: pd.DataFrame,
                               hour: int,
                               primary: str,
                               candidates: List[str]) -> Optional[str]:
    sub = df[df["hours"] == hour]
    if candidates:
        present = set(sub["model"].unique().tolist())
        cand = [c for c in candidates if c in present]
        if cand:
            sub = sub[sub["model"].isin(cand)]
    if sub.empty: return None
    by_model = sub.groupby("model")[primary].mean().reset_index()
    by_model["_key"] = by_model[primary].apply(lambda v: _score_direction(primary, v))
    by_model = by_model.sort_values("_key", kind="mergesort")
    return str(by_model.iloc[0]["model"])

def _pct_impr(current: float, baseline: float, metric: str) -> float:
    if pd.isna(current) or pd.isna(baseline) or baseline == 0:
        return np.nan
    return ((current - baseline) / baseline) * 100.0 if metric == "secs" else ((baseline - current) / baseline) * 100.0

# Small helpers for mix-aware selection
def _best_hour_for_lang_setting(df_all: pd.DataFrame, lang: str, model: str, setting: str, primary: str) -> Optional[int]:
    sub = df_all[(df_all["language"]==lang) & (df_all["model"]==model) & (df_all["train_setting"]==setting)]
    if sub.empty: return None
    g = sub.groupby("hours")[primary].mean().reset_index()
    if g.empty: return None
    g["_key"] = g[primary].apply(lambda v: _score_direction(primary, v))
    g = g.sort_values("_key", kind="mergesort")
    return int(g.iloc[0]["hours"])

def _best_mix_hour_avg_across_langs(df_all: pd.DataFrame, languages: List[str], model: str, primary: str) -> Optional[int]:
    frames = []
    for lang in languages:
        sub = df_all[(df_all["language"]==lang) & (df_all["model"]==model) & (df_all["train_setting"]=="mix")]
        if sub.empty:
            return None
        g = sub.groupby("hours")[primary].mean().rename(f"{lang}_{primary}")
        frames.append(g)
    common = pd.concat(frames, axis=1).dropna()
    if common.empty: return None
    common["avg"] = common.mean(axis=1)
    best_hour = int(common["avg"].idxmin())
    return best_hour

def _mean_metrics(df: pd.DataFrame, metrics: List[str]) -> Dict[str, float]:
    out = {}
    for m in metrics:
        if m in df.columns:
            try:
                if not df.empty and df[m].notna().any():
                    out[m] = float(df[m].mean())
                else:
                    out[m] = np.nan
            except Exception:
                out[m] = np.nan
        else:
            out[m] = np.nan
    return out

# -----------------------------
# RQ1 — component ablation (per language)
# -----------------------------

def rq1_component_ablation(df: pd.DataFrame,
                           figs_dir: Path,
                           tables_dir: Path,
                           language: str,
                           component_models: List[str]) -> Tuple[int, str, pd.DataFrame, Dict[str, Path]]:
    primary = _pick_primary_metric(df)
    winner_setting, best_hour = _select_best_setting_and_hour(df, "full_finetuned", primary)
    if best_hour is None:
        raise RuntimeError("Could not determine best hour for full_finetuned")

    rows = []
    present_models = set(df["model"].unique().tolist())
    ordered_models = [m for m in component_models if m in present_models] or \
                     sorted(df[df["hours"] == best_hour]["model"].dropna().unique().tolist())

    for m in ordered_models:
        sub = df[(df["model"] == m) & (df["hours"] == best_hour)]
        if winner_setting:
            sub = sub[sub["train_setting"].fillna("") == winner_setting]
        if sub.empty:
            continue
        row = {"model": m, "hours": best_hour}
        eff_wer = _effective_wer_col(sub)
        for met in CORE_REPORT_METRICS:
            use = eff_wer if met == "wer" else met
            mu, sd = _agg_mean_std(sub, use)
            row[f"{met}_mean"] = mu
            row[f"{met}_std"]  = sd
        # optionally compute extended metrics for CSV (not shown in LaTeX here to avoid width blow-up)
        for met in EXTENDED_REPORT_METRICS:
            if met in sub.columns:
                mu, sd = _agg_mean_std(sub, met)
                row[f"{met}_mean"] = mu
                row[f"{met}_std"]  = sd
        rows.append(row)

    abl = pd.DataFrame(rows)
    if abl.empty:
        raise RuntimeError("Ablation table is empty; check component models & data availability.")

    baseline_for_delta = "hifigan_only" if "hifigan_only" in abl["model"].values else None
    if baseline_for_delta is not None and "wer_mean" in abl.columns:
        base = abl[abl["model"] == baseline_for_delta].iloc[0]
        abl["delta_wer"] = abl["wer_mean"] - base["wer_mean"]

    # Sort rows by descending WER mean (as requested). NaNs go last.
    if "wer_mean" in abl.columns:
        abl = abl.sort_values("wer_mean", ascending=False, na_position="last", kind="mergesort")

    lang_tag = language.upper()
    rq1_table_path = tables_dir / f"RQ1_component-ablation_{lang_tag}_best{best_hour}h_{winner_setting}-setting_table.tex"
    rq1_csv_path   = tables_dir / f"RQ1_component-ablation_{lang_tag}_best{best_hour}h_{winner_setting}-setting_table.csv"

    abl.to_csv(rq1_csv_path, index=False)

    with open(rq1_table_path, "w") as f:
        # f.write("% auto-generated\n")
        f.write("\\begin{table}[htbp!]\n\\centering\n")
        f.write("\\small\n")
        # Wrap in a resizer to fit linewidth
        f.write("\\resizebox{\\linewidth}{!}{%\n")
        # Only include WER, SECS, MCD (exclude GPE/RTF)
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Component & WER$\\downarrow$ & SECS$\\uparrow$ & MCD$\\downarrow$ \\\\n+\\midrule\n")
        metrics_for_rq1 = ["wer","secs","mcd"]
        for _, r in abl.iterrows():
            cells = []
            for met in metrics_for_rq1:
                mu = r.get(f"{met}_mean", np.nan)
                sd = r.get(f"{met}_std", np.nan)
                key = "wer_norm" if met == "wer" else met
                cells.append(_fmt(mu, sd, key))
            f.write(f"{r['model'].replace('_','+')} & {cells[0]} & {cells[1]} & {cells[2]} \\\n+")
        f.write("\\bottomrule\n\\end{tabular}\n}\n")
        primary_tex = primary.replace('_', r'\\_')
        f.write(
            f"\n\\caption{{Component ablation at best hour ({best_hour}h) for {lang_tag}. "
            f"Primary metric = {primary_tex}. Lower is better except SECS.}}\n"
        )
        f.write("\\label{tab:rq1-ablation-" + language + "}\\n\\end{table}\n")

    rq1_delta_fig_pdf = rq1_delta_fig_png = None
    if "delta_wer" in abl.columns:
        try:
            set_matplotlib_style()
            base_no_ext = figs_dir / f"RQ1_deltaWER-vs-baseline_{lang_tag}_best{best_hour}h_bars"
            fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
            x = np.arange(len(abl))
            ax.bar(x, abl["delta_wer"])
            ax.axhline(0, linestyle="--", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_','+') for m in abl["model"]], rotation=0)
            _finalize_axes(ax, xlabel="Component", ylabel="ΔWER vs baseline (pp)", legend=False)
            rq1_delta_fig_pdf, rq1_delta_fig_png = _save_fig_both(fig, base_no_ext)
        except Exception as e:
            warnings.warn(f"Could not create RQ1 delta plot: {e}")

    meta_path = tables_dir.parent / f"META_best-selection_{lang_tag}.csv"
    pd.DataFrame([{
        "language": lang_tag,
        "primary_metric": primary,
        "winner_setting": winner_setting,
        "best_hour": best_hour
    }]).to_csv(meta_path, index=False)

    artifact_paths = {
        "table_tex": rq1_table_path,
        "table_csv": rq1_csv_path,
        "delta_fig_pdf": rq1_delta_fig_pdf,
        "delta_fig_png": rq1_delta_fig_png,
        "meta_csv": meta_path,
        "best_hour": best_hour,
        "winner_setting": winner_setting,
        "primary_metric": primary,
    }
    return best_hour, winner_setting, abl, artifact_paths

# -----------------------------
# RQ2 — learning curve (primary + SECS)
# -----------------------------

def rq2_learning_curve(df: pd.DataFrame,
                       figs_dir: Path,
                       language: str,
                       winner_setting: str) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    sub = df[(df["model"] == "full_finetuned")]
    if winner_setting:
        sub = sub[sub["train_setting"].fillna("") == winner_setting]
    if sub.empty:
        raise RuntimeError("No rows for full_finetuned in winner setting")

    eff_wer = _effective_wer_col(sub)
    agg_dict = {eff_wer: "mean"}
    if "secs" in sub.columns:
        agg_dict["secs"] = "mean"

    agg = sub.groupby("hours").agg(agg_dict).reset_index().rename(columns={eff_wer: "wer_curve"})
    agg = agg.sort_values("hours")

    set_matplotlib_style()
    lang_tag = language.upper()
    base_no_ext = figs_dir / f"RQ2_learning-curve_{lang_tag}_{winner_setting}-setting_WER-and-SECS_vs-hours_line"
    fig, ax1 = plt.subplots(figsize=(SINGLE_COL_W, 2.4))
    ax2 = ax1.twinx()

    l1, = ax1.plot(agg["hours"], agg["wer_curve"], marker="o", linewidth=1.6,
                   label=("WER-norm" if eff_wer=="wer_norm" else "WER"))
    l2 = None
    if "secs" in agg.columns and agg["secs"].notna().any():
        l2, = ax2.plot(agg["hours"], agg["secs"], marker="s", linewidth=1.2, linestyle="--", label="SECS")

    best_row = agg.loc[agg["wer_curve"].idxmin()]
    ax1.scatter([best_row["hours"]], [best_row["wer_curve"]], s=22, zorder=5)

    _finalize_axes(ax1, xlabel="Training hours",
                   ylabel=("WER (normalized) ↓" if eff_wer == "wer_norm" else "WER ↓"),
                   legend=False)
    # add a compact legend for the two lines
    handles = [l1] + ([l2] if l2 is not None else [])
    labels = [h.get_label() for h in handles]
    if handles:
        ax1.legend(handles, labels, frameon=False, loc="best")
    ax2.set_ylabel("SECS ↑", labelpad=2)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    pdf_path, png_path = _save_fig_both(fig, base_no_ext)
    csv_path = figs_dir / f"RQ2_learning-curve_{lang_tag}_{winner_setting}-setting_WER-and-SECS_vs-hours.csv"
    agg.to_csv(csv_path, index=False)

    # optional alternate artifacts: WER-only and SECS-only (for selection in paper)
    alt_paths: Dict[str, Path] = {}
    try:
        # WER-only
        base_no_ext_w = figs_dir / f"RQ2_learning-curve_{lang_tag}_{winner_setting}-setting_WER_vs-hours_line"
        fig_w, ax_w = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
        ax_w.plot(agg["hours"], agg["wer_curve"], marker="o", linewidth=1.6,
                  label=("WER-norm" if eff_wer=="wer_norm" else "WER"))
        _finalize_axes(ax_w, xlabel="Training hours",
                       ylabel=("WER (normalized) ↓" if eff_wer == "wer_norm" else "WER ↓"),
                       legend=True)
        alt_paths["fig_w_pdf"], alt_paths["fig_w_png"] = _save_fig_both(fig_w, base_no_ext_w)

        # SECS-only (if available)
        if "secs" in agg.columns and agg["secs"].notna().any():
            base_no_ext_s = figs_dir / f"RQ2_learning-curve_{lang_tag}_{winner_setting}-setting_SECS_vs-hours_line"
            fig_s, ax_s = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
            ax_s.plot(agg["hours"], agg["secs"], marker="s", linewidth=1.4, linestyle="--", label="SECS")
            _finalize_axes(ax_s, xlabel="Training hours", ylabel="SECS ↑", legend=True)
            alt_paths["fig_s_pdf"], alt_paths["fig_s_png"] = _save_fig_both(fig_s, base_no_ext_s)
    except Exception as _e:
        warnings.warn(f"Alternate RQ2 plots failed: {_e}")

    return agg, {"fig_pdf": pdf_path, "fig_png": png_path, "csv": csv_path, **alt_paths}

# -----------------------------
# RQ3 — mix vs mono deltas
# -----------------------------

def rq3_mix_vs_mono(df: pd.DataFrame,
                    figs_dir: Path,
                    language: str) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    sub = df[df["model"] == "full_finetuned"]
    mono = sub[sub["train_setting"] == "mono"]
    mix  = sub[sub["train_setting"] == "mix"]
    if mono.empty or mix.empty:
        warnings.warn(f"No matched mono/mix for {language.upper()} — skipping RQ3")
        return pd.DataFrame(), {}

    eff_wer_mono = _effective_wer_col(mono)
    eff_wer_mix  = _effective_wer_col(mix)
    ycol = "wer_norm" if (eff_wer_mono == "wer_norm" and eff_wer_mix == "wer_norm") else "wer"

    hours = sorted(set(mono["hours"].dropna().astype(int)).intersection(
                   set(mix["hours"].dropna().astype(int))))
    rows = []
    for h in hours:
        mA = mono[mono["hours"] == h]
        mB = mix[mix["hours"] == h]
        if mA.empty or mB.empty:
            continue
        row = {"hours": h}
        for met in [ycol,"secs"]:
            if (met in mA.columns) and (met in mB.columns):
                a = mA[met].mean()
                b = mB[met].mean()
                row[f"{met}_mono"] = a
                row[f"{met}_mix"]  = b
                row[f"delta_{met}"] = b - a
        rows.append(row)

    deltas = pd.DataFrame(rows).sort_values("hours")
    if deltas.empty:
        warnings.warn(f"Could not build mix-vs-mono deltas for {language.upper()}")
        return deltas, {}

    set_matplotlib_style()
    lang_tag = language.upper()
    base_no_ext = figs_dir / f"RQ3_mix-vs-mono_deltas_{lang_tag}_WER-and-SECS_stackedbars"
    # Two-panel figure: top ΔWER (or ΔWER-norm), bottom ΔSECS
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(SINGLE_COL_W, 3.6), sharex=True,
                                         gridspec_kw={"hspace": 0.25, "height_ratios": [1.2, 1.0]})
    # Use categorical x positions to ensure visible bar widths regardless of hour spacing
    hrs = deltas["hours"].astype(int).tolist()
    x = np.arange(len(hrs))
    # Top: WER
    ax_top.bar(x, deltas[f"delta_{ycol}"], color="#4C78A8")
    ax_top.axhline(0, linestyle="--", linewidth=0.8)
    _finalize_axes(ax_top, ylabel=f"Δ{('WER-norm' if ycol=='wer_norm' else 'WER')} ↓", legend=False)
    # Bottom: SECS (if present)
    if "delta_secs" in deltas.columns and deltas["delta_secs"].notna().any():
        ax_bot.bar(x, deltas["delta_secs"], color="#F58518")
    else:
        ax_bot.bar(x, [0]*len(hrs), color="#F58518")
    ax_bot.axhline(0, linestyle="--", linewidth=0.8)
    _finalize_axes(ax_bot, xlabel="Training hours (matched)", ylabel="ΔSECS ↑", legend=False)
    # Put original hour labels at the bottom axis
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([str(h) for h in hrs], rotation=0)
    pdf_path, png_path = _save_fig_both(fig, base_no_ext)
    csv_path = figs_dir / f"RQ3_mix-vs-mono_{lang_tag}_deltas.csv"
    deltas.to_csv(csv_path, index=False)

    return deltas, {"fig_pdf": pdf_path, "fig_png": png_path, "csv": csv_path}

# -----------------------------
# B1 — baselines vs best (ROW-WISE)
# -----------------------------

def _nice_baseline_name(m: str) -> str:
    if m == "hifigan_only":
        return "CosyVoice2 (orig.)"
    if m.startswith("baseline_"):
        base = m.replace("baseline_","").lower()
        if ("coqui" in base) or ("xtts" in base):
            return "XTTS2"
        if "elevenlabs" in base:
            return "ElevenLabs"
        if "fishspeech" in base:
            return "Fish Speech"
        return base.replace("_"," ").title()
    return m.replace("_"," ").title()

def b1_baseline_vs_best(df: pd.DataFrame,
                        figs_dir: Path,
                        tables_dir: Path,
                        language: str,
                        candidates: List[str],
                        best_hour: int,
                        winner_setting: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    primary = _pick_primary_metric(df)
    # Lock selection to the RQ1 anchor (winner_setting @ best_hour)
    df_anchor = df.copy()
    if winner_setting:
        df_anchor = df_anchor[df_anchor["train_setting"].fillna("") == winner_setting]
    if df_anchor.empty:
        warnings.warn(f"B1: No data at anchor setting={winner_setting} for {language.upper()}; skipping B1.")
        return pd.DataFrame(), {}

    # Our best model among candidates at the anchor hour and setting
    best_model = _choose_best_model_at_hour(df_anchor, best_hour, primary, candidates) or "full_finetuned"
    ours_sub = df_anchor[(df_anchor["model"] == best_model) & (df_anchor["hours"] == best_hour)]
    if ours_sub.empty:
        warnings.warn(f"B1: Anchor hour/setting present but missing model={best_model} for {language.upper()}; skipping B1.")
        return pd.DataFrame(), {}

    # Baselines include: any 'baseline_*' + 'hifigan_only'
    baseline_models = set(df[df["model"].astype(str).str.startswith("baseline_")]["model"].unique())
    if "hifigan_only" in df["model"].values:
        baseline_models.add("hifigan_only")

    if not baseline_models:
        warnings.warn("No baseline_* or hifigan_only rows found; B1 will be empty.")
        return pd.DataFrame(), {}

    # Determine our metric availability
    eff_wer_ours = _effective_wer_col(ours_sub) if not ours_sub.empty else "wer"

    # Helper: model best hour (its own)
    def _model_best_hour(d: pd.DataFrame, model: str, metric: str) -> Optional[int]:
        sub = d[d["model"] == model]
        if sub.empty: return None
        g = sub.groupby("hours")[metric].mean().reset_index()
        if g.empty: return None
        g["_key"] = g[metric].apply(lambda v: _score_direction(metric, v))
        g = g.sort_values("_key", kind="mergesort")
        return int(g.iloc[0]["hours"])

    # Build paired rows per baseline
    row_records = []
    bars_records = []  # for optional improvement bars
    metrics_for_table = CORE_REPORT_METRICS + [m for m in EXTENDED_REPORT_METRICS if m in df.columns]

    for bname in sorted(baseline_models):
        base_sub_all = df[df["model"] == bname]
        if base_sub_all.empty:
            continue
        eff_wer_base = _effective_wer_col(base_sub_all)
        # model-specific best hour (by its own primary)
        base_best_hour = _model_best_hour(base_sub_all, bname, eff_wer_base)
        if base_best_hour is None:
            continue
        base_sub = base_sub_all[base_sub_all["hours"] == base_best_hour]

        # Aggregate stats
        stats_base = {}
        stats_ours = {}
        for met in metrics_for_table:
            use_b = eff_wer_base if met == "wer" else met
            use_o = eff_wer_ours if met == "wer" else met
            b_mu, b_sd = _agg_mean_std(base_sub, use_b)
            o_mu, o_sd = _agg_mean_std(ours_sub, use_o)
            stats_base[met] = (b_mu, b_sd)
            stats_ours[met] = (o_mu, o_sd)

        # Row 1: Baseline
        row_records.append({
            "pair": _nice_baseline_name(bname),
            "system": _nice_baseline_name(bname),
            "hour": base_best_hour,
            "which": "baseline",
            **{f"{m}_mu": stats_base[m][0] for m in metrics_for_table},
            **{f"{m}_sd": stats_base[m][1] for m in metrics_for_table},
        })

        # Row 2: Ours with deltas (ours − base)
        deltas = {m: (stats_ours[m][0] - stats_base[m][0]) for m in metrics_for_table}
        row_records.append({
            "pair": _nice_baseline_name(bname),
            "system": "Ours (anchored)",
            "hour": best_hour,
            "which": "ours",
            **{f"{m}_mu": stats_ours[m][0] for m in metrics_for_table},
            **{f"{m}_sd": stats_ours[m][1] for m in metrics_for_table},
            **{f"{m}_delta": deltas[m] for m in metrics_for_table},
        })

        # For improvement bars (WER & SECS %)
        bars_records.append({
            "baseline": _nice_baseline_name(bname),
            "wer_impr_pct": _pct_impr(stats_ours["wer"][0], stats_base["wer"][0], "wer"),
            "secs_impr_pct": _pct_impr(stats_ours["secs"][0], stats_base["secs"][0], "secs"),
        })

    if not row_records:
        return pd.DataFrame(), {}

    out_rows = pd.DataFrame(row_records)
    bars_df = pd.DataFrame(bars_records)

    lang_tag = language.upper()
    table_tex = tables_dir / f"B1_baselines-vs-best_ROWWISE_{lang_tag}.tex"
    table_csv = tables_dir / f"B1_baselines-vs-best_ROWWISE_{lang_tag}.csv"
    out_rows.to_csv(table_csv, index=False)

    # Write LaTeX table
    def _fmt_val(mu, sd, metric):
        key = "wer_norm" if metric == "wer" else metric
        return _fmt(mu, sd, key)

    def _fmt_delta(val, metric):
        if pd.isna(val): return ""
        d = DECIMALS.get("wer_norm" if metric=="wer" else metric, 3)
        sign = f"{val:+.{d}f}"
        return f" ({sign})"

    with open(table_tex, "w") as f:
        # f.write("% auto-generated\n")
        # Dynamic columns: System + core + optional extended
        # Exclude GPE and RTF per request
        cols = ["System", "WER$\\downarrow$", "SECS$\\uparrow$", "MCD$\\downarrow$"]
        if "f0_rmse_hz_mu" in out_rows.columns:
            cols.append("F0 RMSE$\\downarrow$")
        if "f0_corr_mu" in out_rows.columns:
            cols.append("F0 Corr$\\uparrow$")
        if "vuv_mu" in out_rows.columns:
            cols.append("V/UV$\\downarrow$")
        colspec = "l" + "c" * (len(cols)-1)

        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        # Wrap into resizer for linewidth fit
        f.write("\\resizebox{\\linewidth}{!}{%\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
        f.write(" & ".join(cols) + " \\\n\\midrule\n")

        # emit pairs
        for pair_name in out_rows["pair"].unique():
            chunk = out_rows[out_rows["pair"] == pair_name].copy()
            chunk = chunk.sort_values("which")  # baseline first, then ours
            # baseline row
            b = chunk[chunk["which"]=="baseline"].iloc[0]
            line = [pair_name]
            for m in ["wer","secs","mcd"]:  # exclude gpe
                line.append(_fmt_val(b[f"{m}_mu"], b[f"{m}_sd"], m))
            for m in ["f0_rmse_hz","f0_corr","vuv"]:  # exclude rtf
                if f"{m}_mu" in out_rows.columns:
                    line.append(_fmt_val(b.get(f"{m}_mu", np.nan), b.get(f"{m}_sd", np.nan), m))
            f.write(" & ".join(line) + " \\\n")

            # ours row with deltas
            o = chunk[chunk["which"]=="ours"].iloc[0]
            f.write("\\midrule\n")
            line = ["Ours (anchored)"]
            for m in ["wer","secs","mcd"]:  # exclude gpe
                cell = _fmt_val(o[f"{m}_mu"], o[f"{m}_sd"], m) + _fmt_delta(o.get(f"{m}_delta", np.nan), m)
                line.append(cell)
            for m in ["f0_rmse_hz","f0_corr","vuv"]:  # exclude rtf
                if f"{m}_mu" in out_rows.columns:
                    cell = _fmt_val(o.get(f"{m}_mu", np.nan), o.get(f"{m}_sd", np.nan), m) + _fmt_delta(o.get(f"{m}_delta", np.nan), m)
                    line.append(cell)
            f.write(" & ".join(line) + " \\\n")

        f.write("\\bottomrule\n\\end{tabular}\n}\n")
        anchor_set = (winner_setting or "mono")
        f.write(f"\\caption{{Baselines vs Ours for {lang_tag}. Ours is evaluated at the RQ1 anchor {best_hour}h ({anchor_set}); each baseline is evaluated at its own best hour. Deltas in parentheses are (Ours − Baseline); negative favors Ours for error-type metrics. WER is normalized.}}\n")
        f.write("\\label{tab:baseline-vs-best-rowwise-" + language + "}\n\\end{table}\n")

    # Optional improvement bars (WER/SECS %)
    set_matplotlib_style()
    base_no_ext = figs_dir / f"B1_improvements-vs-baselines_{lang_tag}_bars"
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
    labels = bars_df["baseline"].tolist()
    x = np.arange(len(labels))
    width = 0.42
    wer_imp = bars_df["wer_impr_pct"] if "wer_impr_pct" in bars_df.columns else pd.Series([np.nan]*len(bars_df))
    secs_imp = bars_df["secs_impr_pct"] if "secs_impr_pct" in bars_df.columns else pd.Series([np.nan]*len(bars_df))
    ax.bar(x - width/2, wer_imp, width, label="WER")
    ax.bar(x + width/2, secs_imp, width, label="SECS")
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    _finalize_axes(ax, xlabel="Baseline", ylabel="% improvement over baseline", legend=True)
    imp_pdf, imp_png = _save_fig_both(fig, base_no_ext)

    return out_rows, {"table_tex": table_tex, "table_csv": table_csv,
                      "imp_fig_pdf": imp_pdf, "imp_fig_png": imp_png}

# -----------------------------
# B1 — compact cross-language table (metric → language; means only)
# -----------------------------

def b1_compact_cross_language(df_all: pd.DataFrame,
                              out_dir: Path,
                              languages: List[str],
                              candidates: List[str],
                              anchors: Optional[Dict[str, Dict[str, Union[int,str]]]] = None) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    langs = [l for l in ["fr","de"] if l in set(s.lower() for s in languages)]
    if len(langs) < 2:
        warnings.warn("B1 compact: needs at least FR and DE; skipping.")
        return pd.DataFrame(), {}

    # Resolve systems: CosyVoice2 original (hifigan_only), XTTS2 baseline (baseline_coqui/xtts if present), Ours (anchored)
    systems = []
    present_models = set(df_all["model"].astype(str).unique().tolist())
    if "hifigan_only" in present_models:
        systems.append("hifigan_only")
    # find Coqui/XTTS baseline
    coqui_name = None
    for m in present_models:
        if m.startswith("baseline_") and (("coqui" in m.lower()) or ("xtts" in m.lower())):
            coqui_name = m
            break
    if coqui_name:
        systems.append(coqui_name)

    elevenlabs_name = None
    for m in present_models:
        if m.startswith("baseline_") and ("elevenlabs" in m.lower()):
            elevenlabs_name = m
            break
    if elevenlabs_name:
        systems.append(elevenlabs_name)

    fishspeech_name = None
    for m in present_models:
        if m.startswith("baseline_") and ("fishspeech" in m.lower()):
            fishspeech_name = m
            break
    if fishspeech_name:
        systems.append(fishspeech_name)

    # Ours placeholder
    systems.append("__OURS__")

    # Decide which metrics to include: core always + any extended metrics that exist and have any non-NaN
    # Exclude GPE and RTF per request
    extra_avail = [m for m in EXTENDED_REPORT_METRICS if (m in df_all.columns and df_all[m].notna().any()) ] # and m != "rtf"
    metrics_to_include = ["wer","secs","mcd"] + extra_avail

    # Helper: get mean for a (model, language) at its own best hour (by effective WER)
    def mean_at_best_hour(dfa: pd.DataFrame, model: str, lang: str) -> Dict[str, float]:
        sub = dfa[(dfa["language"]==lang) & (dfa["model"]==model)]
        if sub.empty:
            return {m: np.nan for m in metrics_to_include}
        eff = _effective_wer_col(sub)
        g = sub.groupby("hours")[eff].mean().reset_index()
        if g.empty:
            return {m: np.nan for m in metrics_to_include}
        g["_key"] = g[eff].apply(lambda v: _score_direction(eff, v))
        best_h = int(g.sort_values("_key").iloc[0]["hours"])
        chunk = sub[sub["hours"]==best_h].copy()
        # Build means explicitly to avoid duplicate/overwritten columns when renaming
        means: Dict[str, float] = {}
        means["wer"] = float(chunk[eff].dropna().mean()) if eff in chunk.columns else np.nan
        for met in [m for m in metrics_to_include if m != "wer"]:
            means[met] = float(chunk[met].dropna().mean()) if met in chunk.columns else np.nan
        return means

    # Helper: ours = best candidate at the language's best full_finetuned hour
    def ours_mean(dfa: pd.DataFrame, lang: str) -> Dict[str,float]:
        sub = dfa[(dfa["language"]==lang)]
        if sub.empty:
            return {m: np.nan for m in metrics_to_include}
        primary = _pick_primary_metric(sub)
        # Anchor: if provided, use (winner_setting, best_hour) from RQ1; else fall back to previous behavior
        setting = None
        best_h: Optional[int] = None
        if anchors and lang in anchors:
            setting = anchors[lang].get("winner_setting")  # type: ignore[assignment]
            best_h = anchors[lang].get("best_hour")  # type: ignore[assignment]
        if best_h is None:
            setting, best_h = _select_best_setting_and_hour(sub, "full_finetuned", primary)
        if best_h is None:
            return {m: np.nan for m in metrics_to_include}
        cand = [c for c in candidates if c in list(sub["model"].unique())]
        sub_anchor = sub.copy()
        if setting:
            sub_anchor = sub_anchor[sub_anchor["train_setting"].fillna("") == setting]
        best_model = _choose_best_model_at_hour(sub_anchor, best_h, primary, cand) or "full_finetuned"
        chunk = sub_anchor[(sub_anchor["model"]==best_model) & (sub_anchor["hours"]==best_h)]
        eff = _effective_wer_col(chunk)
        means: Dict[str, float] = {}
        means["wer"] = float(chunk[eff].dropna().mean()) if eff in chunk.columns else np.nan
        for met in [m for m in metrics_to_include if m != "wer"]:
            means[met] = float(chunk[met].dropna().mean()) if met in chunk.columns else np.nan
        return means

    # Build rows
    records = []
    for sys in systems:
        row = {"system": _nice_baseline_name(sys if sys!="__OURS__" else "ours").replace("Ours", "Ours (anchored)")}
        for lang in langs:
            if sys == "__OURS__":
                mets = ours_mean(df_all, lang)
            else:
                mets = mean_at_best_hour(df_all, sys, lang)
            for met in metrics_to_include:
                row[f"{met}_{lang}"] = mets.get(met, np.nan)
        records.append(row)

    out = pd.DataFrame(records)
    if out.empty:
        return out, {}

    # Save CSV
    csv_path = out_dir / "B1_compact_systems_by_language.csv"
    out.to_csv(csv_path, index=False)

    # LaTeX with metric → language subcolumns, means only
    tex_path = out_dir / "B1_compact_systems_by_language.tex"
    with open(tex_path, "w") as f:
        # f.write("% auto-generated\n")
        f.write("\\begin{table}[htbp!]\n\\centering\n\\footnotesize\n")
        f.write("\\sisetup{detect-weight=true,detect-inline-weight=math,round-mode=places,round-precision=2}\n")
        # columns: System | per-metric (FR, DE)
        colspec = "l" + "SS"*len(metrics_to_include)
        # Wrap in resizer for linewidth fit
        f.write("\\resizebox{\\linewidth}{!}{%\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
        # top header line
        def _metric_label(m: str) -> str:
            if m == "wer": return "WER$\\downarrow$"
            if m == "secs": return "SECS$\\uparrow$"
            if m == "mcd": return "MCD$\\downarrow$"
            if m == "gpe": return "GPE$\\downarrow$"
            if m == "rtf": return "RTF$\\downarrow$"
            if m == "f0_rmse_hz": return "F0 RMSE (Hz)$\\downarrow$"
            if m == "f0_corr": return "F0 Corr$\\uparrow$"
            if m == "vuv": return "V/UV$\\downarrow$"
            return m
        tops = ["System"] + [f"\\multicolumn{{2}}{{c}}{{{_metric_label(lbl)}}}" for lbl in metrics_to_include]
        f.write(" ".join([tops[0], "&", " & ".join(tops[1:]), "\\\\\n"]))
        # cmidrules
        pos = 2
        for _ in metrics_to_include:
            f.write(f"\\cmidrule(lr){{{pos}-{pos+1}}}")
            pos += 2
        f.write("\n")
        # sub header
        sub = [" "] + ["FR & DE" for _ in metrics_to_include]
        f.write(" & ".join(sub) + " \\\\n\\midrule\n")

        # Determine best values per (metric, lang) for bolding
        higher_better = {"secs", "f0_corr"}
        best_val = { (m, l): None for m in metrics_to_include for l in ["fr","de"] }
        for met in metrics_to_include:
            for lang in ["fr","de"]:
                col = f"{met}_{lang}"
                if col in out.columns:
                    series = out[col].astype(float)
                    series = series[series.notna()]
                    if len(series) == 0:
                        continue
                    if met in higher_better:
                        best_val[(met, lang)] = float(series.max())
                    else:
                        best_val[(met, lang)] = float(series.min())

        def _fmt_cell(v: float, metric: str) -> str:
            if pd.isna(v):
                return "N/A"
            dkey = "wer_norm" if metric=="wer" else metric
            return f"{v:.{DECIMALS.get(dkey,2)}f}"

        for _, r in out.iterrows():
            row_cells = [r["system"]]
            # midrule before Ours
            if isinstance(r["system"], str) and r["system"].lower().startswith("ours"):
                f.write("\\midrule\n")
            for met in metrics_to_include:
                for lang in ["fr","de"]:
                    raw = r.get(f"{met}_{lang}", np.nan)
                    cell_str = _fmt_cell(raw, met)
                    # bold if best
                    best = best_val.get((met, lang), None)
                    if best is not None and pd.notna(raw) and np.isfinite(raw) and np.isclose(float(raw), best, rtol=0, atol=1e-12):
                        cell_str = f"\\textbf{{{cell_str}}}"
                    row_cells.append(cell_str)
            f.write(" & ".join(row_cells) + " \\\n")

        f.write("\\bottomrule\n\\end{tabular}\n}\n")
        f.write(
            "\\caption{B1 (compact): Systems at their own best hours, except Ours which is fixed to the RQ1 anchor per language (hour+setting). "
            "Means only (no $\\pm$). Columns are metric $\\rightarrow$ language (FR, DE). Units: WER (\\%), SECS (cosine, unitless), MCD (dB), "
            "F0 RMSE (Hz), F0 Corr (unitless), V/UV (\\%). WER is normalized. XTTS2 receives language explicitly; CosyVoice2 and ElevenLabs infer language.}\n"
        )
        f.write("\\label{tab:b1-compact-crosslang}\n\\end{table}\n")

    return out, {"table_tex": tex_path, "table_csv": csv_path}

# -----------------------------
# Appendix: efficiency (optional)
# -----------------------------

def appendix_efficiency(df: pd.DataFrame,
                        figs_dir: Path,
                        tables_dir: Path,
                        language: str,
                        winner_setting: Optional[str]) -> Dict[str, Optional[Path]]:
    sub = df[df["model"] == "full_finetuned"].copy()
    if winner_setting:
        sub = sub[sub["train_setting"].fillna("") == winner_setting]
    if sub.empty or "rtf" not in sub.columns or sub["rtf"].isna().all():
        warnings.warn("No RTF data available for efficiency appendix.")
        return {}

    set_matplotlib_style()
    rtf_agg = sub.groupby("hours")["rtf"].agg(["mean","std","count"]).reset_index()
    lang_tag = language.upper()
    base_no_ext = figs_dir / f"APPX_efficiency_RTF-vs-hours_{lang_tag}_{winner_setting}-setting_line"
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
    ax.errorbar(rtf_agg["hours"], rtf_agg["mean"], yerr=rtf_agg["std"], marker="o", linewidth=1.4)
    _finalize_axes(ax, xlabel="Training hours", ylabel="RTF (↓ is faster)", legend=False)
    rtf_pdf, rtf_png = _save_fig_both(fig, base_no_ext)
    rtf_csv = figs_dir / f"APPX_efficiency_RTF-vs-hours_{lang_tag}_{winner_setting}-setting.csv"
    rtf_agg.to_csv(rtf_csv, index=False)

    eff_wer = _effective_wer_col(sub)
    best_hour = int(sub.groupby("hours")[eff_wer].mean().idxmin())
    best_chunk = sub[sub["hours"] == best_hour]
    mu, sd = _agg_mean_std(best_chunk, "rtf")
    speed_tex = tables_dir / f"APPX_speed_RTF_best_{lang_tag}_best{best_hour}h_table.tex"
    with open(speed_tex, "w") as f:
        # f.write("% auto-generated\n")
        f.write("\\begin{table}[htbp!]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{lc}\n\\toprule\n")
        f.write("Setting & RTF$\\downarrow$ \\\\\n\\midrule\n")
        f.write(f"Best model @ {best_hour}h & {_fmt(mu, sd, 'rtf')} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\caption{{Inference speed (RTF) for best model at best hour on {lang_tag}.}}\n")
        f.write("\\label{tab:speed-best-" + language + "}\n\\end{table}\n")

    return {"rtf_fig_pdf": rtf_pdf, "rtf_fig_png": rtf_png, "rtf_csv": rtf_csv, "speed_tex": speed_tex}

# -----------------------------
# Supplemental: component curves, heatmaps, pivots, metadata, WIN/LOSS
# -----------------------------

def _supp_component_curves(df: pd.DataFrame,
                           figs_dir: Path,
                           language: str,
                           setting: str,
                           component_models: List[str]):
    sub = df[df["train_setting"].fillna("") == setting]
    if sub.empty: return
    eff_wer = _effective_wer_col(sub)
    set_matplotlib_style()
    lang_tag = language.upper()
    base_no_ext = figs_dir / f"SUPP_component-learning-curves_{lang_tag}_{setting}-setting_primary-vs-hours_line"
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.6))
    for m in [c for c in component_models if c in sub["model"].unique()]:
        g = sub[sub["model"]==m].groupby("hours")[eff_wer].mean().reset_index().sort_values("hours")
        if g.empty: continue
        label = m.replace("_","+")
        ax.plot(g["hours"], g[eff_wer], marker="o", linewidth=1.2, label=label)
    _finalize_axes(ax, xlabel="Training hours",
                   ylabel=("WER (normalized) ↓" if eff_wer=="wer_norm" else "WER ↓"),
                   legend=True)
    _save_fig_both(fig, base_no_ext)

def _supp_component_heatmap(df: pd.DataFrame,
                            figs_dir: Path,
                            language: str,
                            setting: str,
                            component_models: List[str]):
    sub = df[df["train_setting"].fillna("") == setting]
    if sub.empty: return
    eff_wer = _effective_wer_col(sub)
    lang_tag = language.upper()
    pvt = sub[sub["model"].isin(component_models)].pivot_table(
        index="model", columns="hours", values=eff_wer, aggfunc="mean"
    )
    if pvt.empty: return
    pvt_csv = figs_dir.parent / f"SUPP_component-by-hour_{lang_tag}_{setting}.csv"
    pvt.sort_index().to_csv(pvt_csv)

    set_matplotlib_style()
    base_no_ext = figs_dir / f"SUPP_component-by-hour_heatmap_{lang_tag}_{setting}_{'WER-NORM' if eff_wer=='wer_norm' else 'WER'}"
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W*0.6, 2.8))
    data = pvt.values
    im = ax.imshow(data, aspect="auto", cmap=HEATMAP_CMAP, interpolation="nearest")
    ax.set_yticks(np.arange(len(pvt.index)))
    ax.set_yticklabels([r.replace("_","+") for r in pvt.index])
    ax.set_xticks(np.arange(len(pvt.columns)))
    ax.set_xticklabels([str(int(h)) for h in pvt.columns], rotation=0)
    _finalize_axes(ax, xlabel="Training hours",
                   ylabel="Component",
                   legend=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(("WER (norm) ↓" if eff_wer=="wer_norm" else "WER ↓"), rotation=90, labelpad=6)
    _save_fig_both(fig, base_no_ext)

def _supp_mix_vs_mono_winloss_by_component(df: pd.DataFrame,
                                           tables_dir: Path,
                                           language: str,
                                           component_models: List[str]):
    sub = df[df["model"].isin(component_models)].copy()
    if sub.empty:
        return
    eff_wer = _effective_wer_col(sub)
    results = []
    for comp in sorted(sub["model"].unique()):
        mono = sub[(sub["model"]==comp) & (sub["train_setting"]=="mono")]
        mix  = sub[(sub["model"]==comp) & (sub["train_setting"]=="mix")]
        if mono.empty or mix.empty:
            continue
        common_hours = sorted(set(mono["hours"].dropna().astype(int)).intersection(
                              set(mix["hours"].dropna().astype(int))))
        if not common_hours:
            continue
        deltas_primary = []
        deltas_secs = []
        wins=losses=ties=0
        for h in common_hours:
            mA = mono[mono["hours"]==h][eff_wer].mean()
            mB = mix[mix["hours"]==h][eff_wer].mean()
            d  = mB - mA  # negative means mix better (for WER-like)
            deltas_primary.append(d)
            if d < -1e-12: wins += 1
            elif d > 1e-12: losses += 1
            else: ties += 1
            if "secs" in mono.columns and "secs" in mix.columns:
                sA = mono[mono["hours"]==h]["secs"].mean()
                sB = mix[mix["hours"]==h]["secs"].mean()
                deltas_secs.append(sB - sA)
        row = {
            "component": comp,
            "hours_compared": len(common_hours),
            "wins_mix_better": wins,
            "losses_mix_worse": losses,
            "ties": ties,
            f"mean_delta_{'wer_norm' if eff_wer=='wer_norm' else 'wer'}": (np.nan if not deltas_primary else float(np.nanmean(deltas_primary))),
            "mean_delta_secs": (np.nan if not deltas_secs else float(np.nanmean(deltas_secs))),
        }
        results.append(row)

    if not results:
        return

    out = pd.DataFrame(results)
    lang_tag = language.upper()
    csv_path = tables_dir / f"SUPP_mix-vs-mono_winloss_{lang_tag}_by-component.csv"
    out.to_csv(csv_path, index=False)

    with open(tables_dir / f"SUPP_mix-vs-mono_winloss_{lang_tag}_by-component.tex", "w") as f:
        # f.write("% auto-generated\n")
        f.write("\\begin{table}[htbp!]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        f.write("Component & Hours & Wins (mix) & Losses & Ties & Mean ΔPrimary (mix−mono) \\\\\n\\midrule\n")
        for _, r in out.iterrows():
            f.write(f"{r['component'].replace('_','+')} & {int(r['hours_compared'])} & "
                    f"{int(r['wins_mix_better'])} & {int(r['losses_mix_worse'])} & {int(r['ties'])} & "
                    f"{r.filter(like='mean_delta_').drop('mean_delta_secs', errors='ignore').values[0]:+.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\caption{{Mix vs mono wins/losses by component for {lang_tag}. "
                f"Negative ΔPrimary favors mix (WER-like).}}\n")
        f.write("\\label{tab:mix-vs-mono-winloss-" + language + "}\n\\end{table}\n")

# -----------------------------
# Cross-language RQ1 (MIX-AWARE)
# -----------------------------

def _component_to_symbols(component: str) -> Tuple[str, str, str]:
    """Map component name to (LLM, Flow, Voc.) symbols using \\oplus (fine-tuned), \\circ (original), \\ominus (partially trained)."""
    symbol_map = {
        "pretrained": ("\\circ", "\\circ", "\\ominus"),
        "hifigan_only": ("\\circ", "\\circ", "\\circ"),
        "flow_only": ("\\circ", "\\oplus", "\\ominus"),
        "flow_hifigan": ("\\circ", "\\oplus", "\\circ"),
        "llm_flow": ("\\oplus", "\\oplus", "\\ominus"),
        "llm_only": ("\\oplus", "\\circ", "\\ominus"),
        "llm_hifigan": ("\\oplus", "\\circ", "\\circ"),
        "full_finetuned": ("\\oplus", "\\oplus", "\\circ"),
    }
    return symbol_map.get(component, ("\\circ", "\\circ", "\\circ"))

def rq1_cross_language_mixaware(df_all: pd.DataFrame,
                                out_root: Path,
                                languages: List[str],
                                components: List[str],
                                rq1_table_hour: Union[str,int]) -> Dict[str, Path]:
    """
    Build table with column groups:
      FR (mono) | DE (mono) | FR+DE (mix; avg of FR-eval and DE-eval)

    Hour selection:
      If 'best': FR hour = best (full_finetuned, mono, FR); DE hour = best (full_finetuned, mono, DE);
                 MIX hour = best avg across FR+DE for (full_finetuned, mix);
      If <int>:  use that fixed hour for all three groups.

    CSV sidecar also writes FR_mix and DE_mix separately for transparency.
    """
    tables_dir = out_root / "cross_language" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    primary = _pick_primary_metric(df_all)
    # Determine hours
    if isinstance(rq1_table_hour, str) and rq1_table_hour.lower() == "best":
        fr_hour = _best_hour_for_lang_setting(df_all, "fr", "full_finetuned", "mono", primary)
        de_hour = _best_hour_for_lang_setting(df_all, "de", "full_finetuned", "mono", primary)
        mix_hour = _best_mix_hour_avg_across_langs(df_all, ["fr","de"], "full_finetuned", primary)
    else:
        fr_hour = de_hour = mix_hour = int(rq1_table_hour)

    hour_tag = "best" if (isinstance(rq1_table_hour, str) and rq1_table_hour.lower()=="best") else f"{int(rq1_table_hour)}h"

    # Write META file for mix best hour selection
    if isinstance(rq1_table_hour, str) and rq1_table_hour.lower() == "best":
        mix_meta_path = tables_dir.parent / "META_best-selection_MIX.csv"
        pd.DataFrame([{
            "languages": "FR+DE",
            "setting": "mix",
            "primary_metric": primary,
            "best_hour": mix_hour,
            "fr_mono_hour": fr_hour,
            "de_mono_hour": de_hour
        }]).to_csv(mix_meta_path, index=False)

    rows = []
    for comp in components:
        row = {"component": comp}

        # FR (mono @ fr_hour)
        dff = df_all[(df_all["language"]=="fr") & (df_all["model"]==comp) &
                     (df_all["train_setting"]=="mono") & (df_all["hours"]==fr_hour)]
        eff_fr = _effective_wer_col(dff) if not dff.empty else "wer"
        m_fr = {}
        for met in ["wer","secs","mcd","gpe"]:
            use = eff_fr if met == "wer" else met
            mu, _ = _agg_mean_std(dff, use)
            m_fr[met] = mu
        row.update({f"fr_{k}": v for k,v in m_fr.items()})

        # DE (mono @ de_hour)
        dfd = df_all[(df_all["language"]=="de") & (df_all["model"]==comp) &
                     (df_all["train_setting"]=="mono") & (df_all["hours"]==de_hour)]
        eff_de = _effective_wer_col(dfd) if not dfd.empty else "wer"
        m_de = {}
        for met in ["wer","secs","mcd","gpe"]:
            use = eff_de if met == "wer" else met
            mu, _ = _agg_mean_std(dfd, use)
            m_de[met] = mu
        row.update({f"de_{k}": v for k,v in m_de.items()})

        # MIX evaluated on FR and DE (both @ mix_hour)
        dfm_fr = df_all[(df_all["language"]=="fr") & (df_all["model"]==comp) &
                        (df_all["train_setting"]=="mix") & (df_all["hours"]==mix_hour)]
        dfm_de = df_all[(df_all["language"]=="de") & (df_all["model"]==comp) &
                        (df_all["train_setting"]=="mix") & (df_all["hours"]==mix_hour)]
        eff_mfr = _effective_wer_col(dfm_fr) if not dfm_fr.empty else "wer"
        eff_mde = _effective_wer_col(dfm_de) if not dfm_de.empty else "wer"
        m_mix_fr = {}
        m_mix_de = {}
        for met in ["wer","secs","mcd","gpe"]:
            use_fr = eff_mfr if met == "wer" else met
            use_de = eff_mde if met == "wer" else met
            mu_fr, _ = _agg_mean_std(dfm_fr, use_fr)
            mu_de, _ = _agg_mean_std(dfm_de, use_de)
            m_mix_fr[met] = mu_fr
            m_mix_de[met] = mu_de
        mix_avg = {}
        for met in ["wer","secs","mcd","gpe"]:
            vals = [m_mix_fr.get(met, np.nan), m_mix_de.get(met, np.nan)]
            vals = [v for v in vals if not pd.isna(v)]  # filter out NaN values
            mix_avg[met] = np.mean(vals) if vals else np.nan
        row.update({f"mix_avg_{k}": v for k,v in mix_avg.items()})
        row.update({f"mix_fr_{k}": v for k,v in m_mix_fr.items()})
        row.update({f"mix_de_{k}": v for k,v in m_mix_de.items()})
        rows.append(row)

    tab = pd.DataFrame(rows)
    if tab.empty:
        return {}

    # Save detailed CSV (with per-side values) and a clean one for LaTeX
    detailed_csv = tables_dir / f"RQ1_cross-language_components_mix-aware_{hour_tag}__mix_sides.csv"
    tab.to_csv(detailed_csv, index=False)

    # Keep CSV with previous columns for reproducibility (including GPE and averages)
    clean = tab[["component",
                 "fr_wer","fr_secs","fr_mcd","fr_gpe",
                 "de_wer","de_secs","de_mcd","de_gpe",
                 "mix_avg_wer","mix_avg_secs","mix_avg_mcd","mix_avg_gpe",
                 "mix_fr_wer","mix_fr_secs","mix_fr_mcd","mix_fr_gpe",
                 "mix_de_wer","mix_de_secs","mix_de_mcd","mix_de_gpe"]].copy()
    clean_csv = tables_dir / f"RQ1_cross-language_components_mix-aware_{hour_tag}.csv"
    clean.to_csv(clean_csv, index=False)

    # Order rows by descending MIX WER average (requested ordering by WER)
    def _mix_avg_wer(row):
        a = row.get("mix_fr_wer", np.nan)
        b = row.get("mix_de_wer", np.nan)
        vals = [v for v in [a, b] if not pd.isna(v)]
        if not vals:
            return np.nan
        return float(np.mean(vals))
    clean["__mix_wer_avg"] = clean.apply(_mix_avg_wer, axis=1)
    clean = clean.sort_values("__mix_wer_avg", ascending=False, na_position="last").drop(columns=["__mix_wer_avg"])

    # LaTeX - Updated format with shading and delta row
    tex_path = tables_dir / f"RQ1_cross-language_components_mix-aware_{hour_tag}.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp!]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\resizebox{\\linewidth}{!}{%\n")
        f.write("\\sisetup{detect-weight=true,detect-inline-weight=math,round-mode=places,round-precision=2}\n")
        
        # Column spec: 3 component symbol columns + 3 groups of 3 metrics each
        colspec = "ccc" + "SSS" + "SSS" + "ccc"
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
        f.write("\\toprule\n")
        
        # Header with column groups
        f.write(" & & & \\multicolumn{3}{c}{FR (mono)} & \\multicolumn{3}{c}{DE (mono)} & \\multicolumn{3}{c}{FR+DE (mix)} \\\\\n")
        f.write("\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\n")
        f.write("\\multicolumn{1}{c}{LLM} & \\multicolumn{1}{c}{Flow} & \\multicolumn{1}{c}{Voc.} & ")
        f.write("\\multicolumn{1}{c}{WER$\\downarrow$} & \\multicolumn{1}{c}{SECS$\\uparrow$} & \\multicolumn{1}{c}{MCD$\\downarrow$} & ")
        f.write("\\multicolumn{1}{c}{WER$\\downarrow$} & \\multicolumn{1}{c}{SECS$\\uparrow$} & \\multicolumn{1}{c}{MCD$\\downarrow$} & ")
        f.write("\\multicolumn{1}{c}{WER$\\downarrow$} & \\multicolumn{1}{c}{SECS$\\uparrow$} & \\multicolumn{1}{c}{MCD$\\downarrow$} \\\\\n")
        f.write("\\midrule\n")

        def _fmt_cell(val):
            if pd.isna(val): return "N/A"
            return f"{val:.2f}"

        def _fmt_with_bold(val, metric, col_name):
            if pd.isna(val): return "N/A"
            formatted = f"{val:.2f}"
            # Mark best values in bold
            valid_vals = clean[col_name].dropna()
            if not valid_vals.empty:
                if metric == "secs":
                    is_best = abs(val - valid_vals.max()) < 1e-6
                else:
                    is_best = abs(val - valid_vals.min()) < 1e-6
                if is_best:
                    return f"\\textbf{{{formatted}}}"
            return formatted

        # Find baseline row (hifigan_only) for delta calculation
        baseline_row = clean[clean["component"] == "hifigan_only"]
        if baseline_row.empty:
            baseline_row = clean.iloc[0:1]  # fallback to first row
        baseline = baseline_row.iloc[0] if not baseline_row.empty else None

        # Write component rows with shading for baseline and best model
        for idx, (_, r) in enumerate(clean.iterrows()):
            llm, flow, voc = _component_to_symbols(r["component"])
            
            # Apply row shading for baseline and full_finetuned
            shading = ""
            if r["component"] == "hifigan_only" or r["component"] == "full_finetuned":
                shading = "\\rowcolor[gray]{0.9} "
            
            # Format cells with bold for best values
            fr_wer = _fmt_with_bold(r.get("fr_wer", np.nan), "wer", "fr_wer")
            fr_secs = _fmt_with_bold(r.get("fr_secs", np.nan), "secs", "fr_secs")
            fr_mcd = _fmt_with_bold(r.get("fr_mcd", np.nan), "mcd", "fr_mcd")
            
            de_wer = _fmt_with_bold(r.get("de_wer", np.nan), "wer", "de_wer")
            de_secs = _fmt_with_bold(r.get("de_secs", np.nan), "secs", "de_secs")
            de_mcd = _fmt_with_bold(r.get("de_mcd", np.nan), "mcd", "de_mcd")
            
            # Mix values as FR/DE pairs with individual bold formatting
            mix_fr_wer = _fmt_with_bold(r.get('mix_fr_wer', np.nan), "wer", "mix_fr_wer")
            mix_de_wer = _fmt_with_bold(r.get('mix_de_wer', np.nan), "wer", "mix_de_wer")
            mix_fr_secs = _fmt_with_bold(r.get('mix_fr_secs', np.nan), "secs", "mix_fr_secs")
            mix_de_secs = _fmt_with_bold(r.get('mix_de_secs', np.nan), "secs", "mix_de_secs")
            mix_fr_mcd = _fmt_with_bold(r.get('mix_fr_mcd', np.nan), "mcd", "mix_fr_mcd")
            mix_de_mcd = _fmt_with_bold(r.get('mix_de_mcd', np.nan), "mcd", "mix_de_mcd")
            
            wer_pair = f"{mix_fr_wer}/{mix_de_wer}"
            secs_pair = f"{mix_fr_secs}/{mix_de_secs}"
            mcd_pair = f"{mix_fr_mcd}/{mix_de_mcd}"
            
            cells = [llm, flow, voc, fr_wer, fr_secs, fr_mcd, de_wer, de_secs, de_mcd, wer_pair, secs_pair, mcd_pair]
            f.write(f"{shading}" + " & ".join(cells) + " \\\\\n")

        # Add delta row (percentage improvement vs baseline)
        if baseline is not None:
            # Find the best model (full_finetuned) for comparison
            best_row = clean[clean["component"] == "full_finetuned"]
            if not best_row.empty:
                best = best_row.iloc[0]
                
                f.write("\\midrule\n")
                f.write("\\addlinespace\n")
                f.write("\\multicolumn{3}{r}{\\footnotesize\\textbf{$\\Delta$ vs baseline}} &\n")
                
                def _calc_delta_pct(best_val, baseline_val, metric):
                    if pd.isna(best_val) or pd.isna(baseline_val) or baseline_val == 0:
                        return "N/A"
                    if metric == "secs":
                        pct = ((best_val - baseline_val) / baseline_val) * 100
                    else:
                        pct = ((baseline_val - best_val) / baseline_val) * 100
                    return f"{pct:+.0f}\\%"
                
                # FR deltas
                fr_wer_delta = _calc_delta_pct(best.get("fr_wer"), baseline.get("fr_wer"), "wer")
                fr_secs_delta = _calc_delta_pct(best.get("fr_secs"), baseline.get("fr_secs"), "secs")
                fr_mcd_delta = _calc_delta_pct(best.get("fr_mcd"), baseline.get("fr_mcd"), "mcd")
                
                # DE deltas
                de_wer_delta = _calc_delta_pct(best.get("de_wer"), baseline.get("de_wer"), "wer")
                de_secs_delta = _calc_delta_pct(best.get("de_secs"), baseline.get("de_secs"), "secs")
                de_mcd_delta = _calc_delta_pct(best.get("de_mcd"), baseline.get("de_mcd"), "mcd")
                
                # Mix deltas as FR/DE pairs
                mix_fr_wer_delta = _calc_delta_pct(best.get("mix_fr_wer"), baseline.get("mix_fr_wer"), "wer")
                mix_de_wer_delta = _calc_delta_pct(best.get("mix_de_wer"), baseline.get("mix_de_wer"), "wer")
                mix_fr_secs_delta = _calc_delta_pct(best.get("mix_fr_secs"), baseline.get("mix_fr_secs"), "secs")
                mix_de_secs_delta = _calc_delta_pct(best.get("mix_de_secs"), baseline.get("mix_de_secs"), "secs")
                mix_fr_mcd_delta = _calc_delta_pct(best.get("mix_fr_mcd"), baseline.get("mix_fr_mcd"), "mcd")
                mix_de_mcd_delta = _calc_delta_pct(best.get("mix_de_mcd"), baseline.get("mix_de_mcd"), "mcd")
                
                wer_delta_pair = f"\\text{{{mix_fr_wer_delta}}}/\\text{{{mix_de_wer_delta}}}"
                secs_delta_pair = f"\\text{{{mix_fr_secs_delta}}}/\\text{{{mix_de_secs_delta}}}"
                mcd_delta_pair = f"\\text{{{mix_fr_mcd_delta}}}/\\text{{{mix_de_mcd_delta}}}"
                
                f.write("\\multicolumn{1}{c}{\\textcolor{mildgreen}{" + fr_wer_delta + "}} &\n")
                f.write("\\multicolumn{1}{c}{\\textcolor{mildgreen}{" + fr_secs_delta + "}} &\n")
                f.write("\\multicolumn{1}{c}{\\textcolor{mildgreen}{" + fr_mcd_delta + "}} &\n")
                f.write("\\multicolumn{1}{c}{\\textcolor{mildgreen}{" + de_wer_delta + "}} &\n")
                f.write("\\multicolumn{1}{c}{\\textcolor{mildgreen}{" + de_secs_delta + "}} &\n")
                f.write("\\multicolumn{1}{c}{\\textcolor{mildgreen}{" + de_mcd_delta + "}} &\n")
                f.write("\\multicolumn{1}{c}{\\textcolor{mildgreen}{" + wer_delta_pair + "}} &\n")
                f.write("\\multicolumn{1}{c}{\\textcolor{mildgreen}{" + secs_delta_pair + "}} &\n")
                f.write("\\multicolumn{1}{c}{\\textcolor{mildgreen}{" + mcd_delta_pair + "}}\n")
                f.write("\\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}% end resizebox\n")
        
        # Table notes and caption
        f.write("\\begin{tablenotes}[flushleft]\n")
        f.write("\\footnotesize\n")
        f.write("\\item \\textbf{Symbols:} $\\circ$ original CosyVoice2 (unaltered); $\\ominus$ partially trained; $\\oplus$ fine-tuned for French and German.\n")
        f.write("\\item \\textbf{Shading:} gray rows mark the original CosyVoice2 baseline and our fully fine-tuned model.\n")
        f.write("\\end{tablenotes}\n")
        
        if isinstance(rq1_table_hour, str) and rq1_table_hour.lower()=="best":
            f.write("\\caption{Cross-language component table at each group's best hour: FR/DE (mono) use their own best hours; FR+DE (mix) uses the mix model's best hour. Mix values are shown as FR/DE.}\n")
        else:
            f.write(f"\\caption{{Cross-language component table at fixed {int(rq1_table_hour)}h. FR/DE (mono) and FR+DE (mix) all evaluated at the same hour. Mix values are shown as FR/DE.}}\n")
        f.write("\\label{tab:rq1-cross-language-mixaware}\n")
        f.write("\\end{threeparttable}\n")
        f.write("\\end{table}\n")

    return {"table_tex": tex_path, "table_csv": clean_csv, "mix_sides_csv": detailed_csv}

# -----------------------------
# Driver
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Story-first CosyVoice2 report builder (v4.2)")
    ap.add_argument("--results_dir", required=True, help="Directory with *_metrics*.csv files")
    ap.add_argument("--output_dir", default="report_out", help="Where to write figures/tables")
    ap.add_argument("--languages", default="", help="Comma-separated (e.g., fr,de). Empty=auto infer from data")
    ap.add_argument("--backbone", default="", help="Optional filter by backbone exact string")
    ap.add_argument("--components", default="", help="Override component model list (comma-separated)")
    # Single switch for all extras
    ap.add_argument("--supplemental", action="store_true",
                    help="Emit cross-language RQ1, extra CSVs, RTF efficiency, component curves/heatmaps, and mix-vs-mono win/loss")
    # Keep hour selection simple: 'best' (default) or a fixed integer
    ap.add_argument("--rq1_table_hour", default="best", help="'best' or an integer hour like 1500 (applies to cross-language when supplemental enabled)")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df_all = load_all_results(args.results_dir, language=None, backbone=(args.backbone or None))
    langs = [s.strip().lower() for s in args.languages.split(",") if s.strip()] or \
            sorted(list(df_all["language"].dropna().unique()) or ["fr","de"])
    # If user overrides --components we respect it for all steps. Otherwise:
    # - RQ1 uses an extended set to showcase ablations and baseline provenance
    # - The rest (RQ2/RQ3/B1/etc.) uses the main finetuned-only set
    user_components = [s.strip() for s in args.components.split(",") if s.strip()]
    components_rq1 = user_components or (RQ1_EXTRA_COMPONENTS + MAIN_COMPONENT_MODELS)
    # preserve order and uniqueness while keeping RQ1 extras first
    seen = set()
    components_rq1 = [m for m in components_rq1 if not (m in seen or seen.add(m))]
    components_main = user_components or MAIN_COMPONENT_MODELS

    key_figs_dir = out_root / "key_artifacts" / "figs"
    key_tabs_dir = out_root / "key_artifacts" / "tables"
    key_figs_dir.mkdir(parents=True, exist_ok=True)
    key_tabs_dir.mkdir(parents=True, exist_ok=True)

    readme_lines = ["# Generated artifacts\n"]

    per_lang_meta = {}
    for lang in langs:
        lang_dir = out_root / lang
        figs_dir   = lang_dir / "figs"
        tables_dir = lang_dir / "tables"
        figs_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        df = df_all[df_all["language"] == lang]
        if df.empty:
            warnings.warn(f"No rows for language={lang}; skipping.")
            continue

        # RQ1
        best_hour, winner_setting, abl, rq1_paths = rq1_component_ablation(df, figs_dir, tables_dir, lang, components_rq1)
        per_lang_meta[lang] = {"best_hour": rq1_paths["best_hour"],
                               "winner_setting": rq1_paths["winner_setting"],
                               "primary_metric": rq1_paths["primary_metric"]}

        # RQ2
        lc_df, rq2_paths = rq2_learning_curve(df, figs_dir, lang, rq1_paths["winner_setting"])

        # RQ3
        r3_df, rq3_paths = rq3_mix_vs_mono(df, figs_dir, lang)

        # B1 (row-wise) — lock to RQ1 anchor
        b1_df, b1_paths = b1_baseline_vs_best(df, figs_dir, tables_dir, lang, components_main, best_hour, winner_setting)

        # Appendix & supplemental material
        if args.supplemental:
            # Component diagnostics
            _supp_component_curves(df, figs_dir, lang, rq1_paths["winner_setting"], components_rq1)
            _supp_component_heatmap(df, figs_dir, lang, rq1_paths["winner_setting"], components_rq1)
            _supp_mix_vs_mono_winloss_by_component(df, tables_dir, lang, components_rq1)

            # Extra CSV aggregates (duration curves)
            dur = df[df["model"] == "full_finetuned"].copy()
            if rq1_paths["winner_setting"]:
                dur = dur[dur["train_setting"].fillna("") == rq1_paths["winner_setting"]]
            agg_cols = {}
            for met in ["wer_norm","wer","secs","mcd","gpe","rtf","f0_rmse_hz","f0_corr","vuv"]:
                if met in dur.columns:
                    agg_cols[met] = "mean"
            if agg_cols:
                dur_agg = dur.groupby(["train_setting","hours"]).agg(agg_cols).reset_index()
                dur_agg.to_csv(lang_dir / f"APPX_duration-aggregates_{lang.upper()}_{rq1_paths['winner_setting']}-setting.csv", index=False)

            # Efficiency appendix
            appendix_efficiency(df, figs_dir, tables_dir, lang, rq1_paths["winner_setting"])

        # README entries
        readme_lines.append(f"\n## {lang.upper()}")
        if rq1_paths.get("table_tex"):
            readme_lines.append(f"- tables/{Path(rq1_paths['table_tex']).name}")
        if rq1_paths.get("delta_fig_pdf"):
            readme_lines.append(f"- figs/{Path(rq1_paths['delta_fig_pdf']).name} (optional)")
        if rq2_paths.get("fig_pdf"):
            readme_lines.append(f"- figs/{Path(rq2_paths['fig_pdf']).name}")
        if rq3_paths.get("fig_pdf"):
            readme_lines.append(f"- figs/{Path(rq3_paths['fig_pdf']).name}")
        if b1_paths.get("table_tex"):
            readme_lines.append(f"- tables/{Path(b1_paths['table_tex']).name} (row-wise)")
        if b1_paths.get("imp_fig_pdf"):
            readme_lines.append(f"- figs/{Path(b1_paths['imp_fig_pdf']).name} (optional)")
        if args.supplemental:
            readme_lines.append(f"- figs/SUPP_component-learning-curves_{lang.upper()}_{rq1_paths['winner_setting']}-setting_primary-vs-hours_line.pdf")
            readme_lines.append(f"- figs/SUPP_component-by-hour_heatmap_{lang.upper()}_{rq1_paths['winner_setting']}_{rq1_paths['primary_metric'].upper().replace('_','-')}.pdf")
            readme_lines.append(f"- SUPP_component-by-hour_{lang.upper()}_{rq1_paths['winner_setting']}.csv")
            readme_lines.append(f"- tables/SUPP_mix-vs-mono_winloss_{lang.upper()}_by-component.csv (win/loss)")
            readme_lines.append(f"- tables/SUPP_mix-vs-mono_winloss_{lang.upper()}_by-component.tex (win/loss)")
            readme_lines.append(f"- figs/APPX_efficiency_RTF-vs-hours_{lang.upper()}_{rq1_paths['winner_setting']}-setting_line.pdf (appendix, if RTF)")
            readme_lines.append(f"- tables/APPX_speed_RTF_best_{lang.upper()}_best{best_hour}h_table.tex (appendix, if RTF)")
            readme_lines.append(f"- APPX_duration-aggregates_{lang.upper()}_{rq1_paths['winner_setting']}-setting.csv (appendix)")

        # Collect key artifacts
        try:
            if rq1_paths.get("table_tex"):
                shutil.copy2(rq1_paths["table_tex"], key_tabs_dir / Path(rq1_paths["table_tex"]).name)
            if b1_paths.get("table_tex"):
                shutil.copy2(b1_paths["table_tex"], key_tabs_dir / Path(b1_paths["table_tex"]).name)
            if rq2_paths.get("fig_pdf"):
                shutil.copy2(rq2_paths["fig_pdf"], key_figs_dir / Path(rq2_paths["fig_pdf"]).name)
            if rq3_paths.get("fig_pdf"):
                shutil.copy2(rq3_paths["fig_pdf"], key_figs_dir / Path(rq3_paths["fig_pdf"]).name)
        except Exception as e:
            warnings.warn(f"Failed to copy key artifacts for {lang.upper()}: {e}")

    # Cross-language MIX-AWARE table (only when supplemental on)
    if args.supplemental:
        try:
            hour_sel = args.rq1_table_hour
            if isinstance(hour_sel, str) and hour_sel.lower() != "best":
                hour_sel = int(hour_sel)
            xlang_paths = rq1_cross_language_mixaware(df_all, out_root, langs, components_rq1, hour_sel)
            if isinstance(xlang_paths, dict) and xlang_paths.get("table_tex"):
                readme_lines.append("\n## CROSS-LANGUAGE (MIX-AWARE)")
                readme_lines.append(f"- cross_language/tables/{Path(xlang_paths['table_tex']).name}")
                readme_lines.append(f"- cross_language/tables/{Path(xlang_paths['table_csv']).name}")
                readme_lines.append(f"- cross_language/tables/{Path(xlang_paths['mix_sides_csv']).name} (mix FR/DE details)")
                try:
                    shutil.copy2(xlang_paths["table_tex"], key_tabs_dir / Path(xlang_paths["table_tex"]).name)
                except Exception as ce:
                    warnings.warn(f"Could not collect cross-language key artifact: {ce}")
        except Exception as e:
            warnings.warn(f"Cross-language table failed: {e}")

    # B1 compact cross-language (key artifact)
    try:
        # Build anchors map from per_lang_meta
        anchors = {k: {"best_hour": v.get("best_hour"), "winner_setting": v.get("winner_setting")} for k,v in per_lang_meta.items()}
        b1c_df, b1c_paths = b1_compact_cross_language(df_all, key_tabs_dir, langs, components_main, anchors)
        if b1c_paths.get("table_tex"):
            readme_lines.append("\n## B1 (COMPACT)")
            readme_lines.append(f"- key_artifacts/tables/{Path(b1c_paths['table_tex']).name}")
            readme_lines.append(f"- key_artifacts/tables/{Path(b1c_paths['table_csv']).name}")
    except Exception as e:
        warnings.warn(f"B1 compact table failed: {e}")

    # README
    with open(out_root / "README.md", "w") as f:
        f.write("\n".join(readme_lines) + "\n")

    print("\nDone. Paper-ready artifacts (PDF+PNG) written under per-language folders.")
    print(f"Key artifacts collected in: {key_figs_dir.parent}")
    print("See README.md for file list.\n")

if __name__ == "__main__":
    main()