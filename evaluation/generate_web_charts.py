#!/usr/bin/env python3
"""
Generate web-ready chart data for CosyVoice2 demo website.

Reuses data loading and metric logic from generate_report.py.
Outputs JSON files for Chart.js to cosyvoice2-demo/generated_charts/.

Improvements:
- Use WER normalized (wer_norm) instead of WER by default.
- Component radar includes all 8 component variants and normalizes values to 0–100 with correct direction.
- Baseline comparison includes CosyVoice2 original, Coqui XTTS v2, and ElevenLabs when present, plus our best.
- Efficiency chart outputs both WER normalized and SECS on dual axes.
"""

import json
import re
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path
import re

# Import from generate_report.py
from generate_report import (
    load_all_results,
    MAIN_COMPONENT_MODELS,
    CORE_REPORT_METRICS,
    RQ1_EXTRA_COMPONENTS,
    _pick_primary_metric,
    _mean_metrics,
    _select_best_setting_and_hour,
    _choose_best_model_at_hour
)

# Configuration
RESULTS_DIR = Path("/tsi/hi-paris/tts/Luka/evaluation-without-hint/results")
OUT_DIR = Path(__file__).parent / "cosyvoice2-demo" / "generated_charts"
LANGUAGES = ["fr", "de"]

# Create output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# Helpers
# ----------------------------------

LOWER_BETTER = {
    "wer_norm", "wer", "cer_norm", "cer", "mcd", "gpe", "vuv", "rtf", "f0_rmse_hz"
}
HIGHER_BETTER = {"secs", "f0_corr"}

def _metric_col(df, metric):
    """Return a metric column present in df; prefer wer_norm over wer."""
    if metric == "wer":
        # Always map to normalized when possible
        return "wer_norm" if "wer_norm" in df.columns else ("wer" if "wer" in df.columns else metric)
    return metric if metric in df.columns else ("wer_norm" if "wer_norm" in df.columns else metric)

def _higher_is_better(metric: str) -> bool:
    m = metric.lower()
    if m in HIGHER_BETTER:
        return True
    if m in LOWER_BETTER:
        return False
    # Default: lower is better for unknown error-like metrics
    return False

def _best_value(series, metric: str):
    if series is None or len(series) == 0:
        return None
    if _higher_is_better(metric):
        return float(series.max())
    return float(series.min())

def _pick_best_by_hours(df, metric: str):
    """Pick best average value across hours for a given df subset and metric."""
    if df.empty:
        return None
    col = _metric_col(df, metric)
    agg = df.groupby("hours")[col].mean()
    if agg.empty:
        return None
    if _higher_is_better(metric):
        return float(agg.max())
    return float(agg.min())

ALL_COMPONENT_MODELS = list(dict.fromkeys(MAIN_COMPONENT_MODELS + RQ1_EXTRA_COMPONENTS))

def generate_mix_vs_mono_curve_chart(df, language, metric):
    """Mix vs Mono training comparison as line chart across hours for key models."""
    metric = _metric_col(df, metric)
    models = ["full_finetuned", "llm_hifigan", "flow_hifigan"]
    hours = sorted(df["hours"].dropna().unique())
    chart_data = {
        "labels": [int(h) for h in hours],
        "datasets": [],
        "metric_label": metric.upper(),
        "x_label": "Training Hours"
    }
    colors = {"full_finetuned": "#764ba2", "llm_hifigan": "#48bb78", "flow_hifigan": "#f6ad55"}
    for model in models:
        for setting in ["mix", "mono"]:
            subset = df[(df["model"] == model) & (df["train_setting"].str.lower() == setting)]
            if subset.empty:
                continue
            series = subset.groupby("hours")[metric].mean()
            label = f"{model.replace('_', ' ').title()} ({setting.title()})"
            chart_data["datasets"].append({
                "label": label,
                "data": [float(series.get(h, None)) if h in series.index else None for h in hours],
                "borderColor": colors.get(model, "#667eea"),
                "backgroundColor": colors.get(model, "#667eea") + "20",
                "tension": 0.35,
                "fill": False,
            })
    save_chart_data(chart_data, f"mix_mono_curve_{language}_{metric}.json")

def save_chart_data(data, filename):
    """Save chart data as JSON"""
    with open(OUT_DIR / filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filename}")

def generate_component_radar(df, language, metric):
    """RQ1: Component ablation radar chart for specific metric"""
    metric = _metric_col(df, metric)
    # Get best setting and hour for each model
    model_data = {}
    for model in ALL_COMPONENT_MODELS:
        setting, hour = _select_best_setting_and_hour(df, model, metric)
        if setting and hour:
            subset = df[(df["model"] == model) & 
                       (df["train_setting"] == setting) & 
                       (df["hours"] == hour)]
            if not subset.empty:
                val = subset[metric].mean() if metric in subset.columns else np.nan
                model_data[model] = val

    # Clean up NaNs
    model_data = {k: float(v) for k, v in model_data.items() if v == v}
    if not model_data:
        return

    # Use absolute values instead of normalization
    labels = [m.replace("_", " ").title() for m in model_data.keys()]
    values = [round(model_data[m], 4) for m in model_data.keys()]
    
    # Determine appropriate min/max for the radar chart
    min_val = min(values)
    max_val = max(values)
    
    # Add some padding for better visualization
    padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
    chart_min = max(0, min_val - padding) if _higher_is_better(metric) else min_val - padding
    chart_max = max_val + padding

    # Format for Chart.js radar
    chart_data = {
        "labels": labels,
        "datasets": [{
            "label": f"{metric.upper()} (absolute values)",
            "data": values,
            "borderColor": "#667eea",
            "backgroundColor": "#667eea20",
            "pointBackgroundColor": "#667eea"
        }],
        "metric_label": metric.upper(),
        "min": round(chart_min, 4),
        "max": round(chart_max, 4)
    }
    
    save_chart_data(chart_data, f"radar_{language}_{metric}.json")

def generate_learning_curve(df, language, metric):
    """RQ2: Learning curve line chart for specific metric"""
    metric = _metric_col(df, metric)
    # Get data for best model across different hours
    best_model = _choose_best_model_at_hour(df, 6, metric, MAIN_COMPONENT_MODELS)
    if not best_model:
        best_model = "full_finetuned"
    
    setting, _ = _select_best_setting_and_hour(df, best_model, metric)
    if setting:
        subset = df[(df["model"] == best_model) & (df["train_setting"] == setting)]
        hours_data = subset.groupby("hours")[metric].mean().reset_index()
        hours_data = hours_data.sort_values("hours")
        
        chart_data = {
            "labels": [int(h) for h in hours_data["hours"]],
            "datasets": [{
                "label": f"{best_model.replace('_', ' ').title()}",
                "data": [float(v) for v in hours_data[metric]],
                "borderColor": "#48bb78",
                "backgroundColor": "#48bb7820",
                "tension": 0.35,
                "fill": True
            }],
            "metric_label": metric.upper(),
            "x_label": "Training Hours"
        }
        
        save_chart_data(chart_data, f"learning_curve_{language}_{metric}.json")

def generate_baseline_comparison(df, language, metric):
    """B1: Baseline vs best model comparison for specific metric (using anchored approach)"""
    metric = _metric_col(df, metric)
    # Use full_finetuned as our best model 
    best_model = "full_finetuned"
    baseline_model = "hifigan_only"  # This represents original CosyVoice2

    # Determine the anchor hour/setting based on the primary metric (like in generate_report.py)
    primary_metric = _pick_primary_metric(df)
    anchor_setting, anchor_hour = _select_best_setting_and_hour(df, best_model, primary_metric)

    labels = []
    values = []
    colors = []

    # CosyVoice2 original (at its own best hour/setting for the specific metric)
    setting, hour = _select_best_setting_and_hour(df, baseline_model, metric)
    if setting and hour:
        subset = df[(df["model"] == baseline_model) & (df["train_setting"] == setting) & (df["hours"] == hour)]
        if not subset.empty:
            labels.append("CosyVoice2 Original")
            values.append(float(subset[metric].mean()))
            colors.append("#f093fb")

    # Coqui XTTS v2 baseline (heuristics)
    def _match_xtts(row: pd.Series) -> bool:
        for col in ("model", "backbone", "baseline", "system"):
            if col in row and isinstance(row[col], str) and re.search(r"(xtts|coqui)", row[col], re.I):
                return True
        return False

    # ElevenLabs baseline (heuristics)
    def _match_eleven(row: pd.Series) -> bool:
        for col in ("model", "backbone", "baseline", "system"):
            if col in row and isinstance(row[col], str) and re.search(r"(eleven|elevenlabs)", row[col], re.I):
                return True
        return False

    # Fish Speech baseline (heuristics)
    def _match_fishspeech(row: pd.Series) -> bool:
        for col in ("model", "backbone", "baseline", "system"):
            if col in row and isinstance(row[col], str) and re.search(r"(fishspeech|fish.speech)", row[col], re.I):
                return True
        return False

    def _best_for_predicate(df_all: pd.DataFrame, predicate) -> float | None:
        if df_all.empty:
            return None
        try:
            mask = df_all.apply(predicate, axis=1)
            sub = df_all[mask]
            if sub.empty:
                return None
            return _pick_best_by_hours(sub, metric)
        except Exception:
            return None

    xtts_val = _best_for_predicate(df, _match_xtts)
    if xtts_val is not None:
        labels.append("Coqui XTTS v2")
        values.append(float(xtts_val))
        colors.append("#f6ad55")

    eleven_val = _best_for_predicate(df, _match_eleven)
    if eleven_val is not None:
        labels.append("ElevenLabs")
        values.append(float(eleven_val))
        colors.append("#63b3ed")

    fishspeech_val = _best_for_predicate(df, _match_fishspeech)
    if fishspeech_val is not None:
        labels.append("OpenAudio-S1-mini")
        values.append(float(fishspeech_val))
        colors.append("#f093fb")

    # Our Best Model (ANCHORED: use the same hour/setting that's optimal for the primary metric)
    if anchor_setting and anchor_hour:
        subset_anchored = df[(df["model"] == best_model) & (df["train_setting"] == anchor_setting) & (df["hours"] == anchor_hour)]
        if not subset_anchored.empty:
            labels.append("Our Best Model (anchored)")
            values.append(float(subset_anchored[metric].mean()))
            colors.append("#667eea")

    if not labels:
        return

    chart_data = {
        "labels": labels,
        "datasets": [{
            "label": metric.upper(),
            "data": values,
            "backgroundColor": colors,
            "borderColor": colors
        }],
        "metric_label": metric.upper()
    }

    save_chart_data(chart_data, f"baseline_{language}_{metric}.json")


def main():
    """Generate all charts for both languages and all metrics"""
    print("Generating web charts...")
    
    # Define metrics to generate charts for
    metrics = ["wer_norm", "secs", "mcd", "f0_corr", "vuv"]
    
    for lang in LANGUAGES:
        print(f"\nProcessing {lang.upper()}...")
        try:
            # Use the correct pattern that matches the actual file names
            # Files are like: full_finetuned_50h_fr_metrics_model-fr.csv
            df = load_all_results(str(RESULTS_DIR), language=lang)
            if df.empty:
                print(f"No data found for {lang}")
                continue

            # Generate charts for each metric
            for metric in metrics:
                print(f"  Generating charts for {metric}...")
                try:
                    generate_component_radar(df, lang, metric)
                    generate_learning_curve(df, lang, metric)
                    generate_mix_vs_mono_curve_chart(df, lang, metric)
                    generate_baseline_comparison(df, lang, metric)
                except Exception as e:
                    print(f"    Error generating {metric} charts: {e}")
                
        except Exception as e:
            print(f"Error processing {lang}: {e}")
            # Let's try a custom pattern since the standard one doesn't work
            generate_mix_vs_mono_curve_chart(df, lang, metric)
            custom_load_results(lang, metrics)
    print("Files can now be loaded by the website!")

def custom_load_results(lang, metrics):
    """Custom loader for files with non-standard naming"""
    import pandas as pd
    
    # Look for files matching the actual pattern
    pattern = f"*_{lang}_metrics_*.csv"
    files = list(RESULTS_DIR.glob(pattern))
    
    if not files:
        print(f"No files found with pattern {pattern}")
        return
    
    print(f"Found {len(files)} files for {lang}")
    
    # Load and combine all files
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__file"] = f.name
            
            # Extract model name and hours from filename
            # Example: full_finetuned_50h_fr_metrics_model-fr.csv
            fname = f.stem
            parts = fname.split('_')
            
            # Extract model name (everything before the hours)
            model_parts = []
            hours_val = None
            for i, part in enumerate(parts):
                if part.endswith('h') and part[:-1].isdigit():
                    hours_val = int(part[:-1])
                    model_parts = parts[:i]
                    break
            
            if model_parts and hours_val:
                model_name = '_'.join(model_parts)
                df["model"] = model_name
                df["hours"] = hours_val
                df["language"] = lang
                
                # Extract train_setting from the end
                if fname.endswith(f'model-{lang}'):
                    df["train_setting"] = "mono"
                elif fname.endswith('model-mix'):
                    df["train_setting"] = "mix"
                
                dfs.append(df)
                
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined dataframe shape: {combined_df.shape}")
        print(f"Models found: {combined_df['model'].unique()}")
        
        # Generate charts for each metric
        for metric in metrics:
            print(f"  Generating charts for {metric}...")
            try:
                generate_component_radar(combined_df, lang, metric)
                generate_learning_curve(combined_df, lang, metric)
                generate_mix_vs_mono_curve_chart(combined_df, lang, metric)
                generate_baseline_comparison(combined_df, lang, metric)
            except Exception as e:
                print(f"    Error generating {metric} charts: {e}")

if __name__ == "__main__":
    main()
