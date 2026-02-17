#!/usr/bin/env python3
import argparse
import datetime as dt
import glob
import os
from typing import Dict, List, Optional

import pandas as pd


def read_env_file(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    data: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if "#" in v:
                v = v.split("#", 1)[0].strip()
            data[k.strip()] = v.strip()
    return data


def f6(val: Optional[float]) -> str:
    if val is None or pd.isna(val):
        return "n/a"
    return f"{float(val):.6f}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_no rows_"
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header_line, sep_line] + body)


def first_round_at_or_above(df: pd.DataFrame, metric: str, threshold: float) -> Optional[int]:
    q = df[df[metric] >= threshold]
    if q.empty:
        return None
    return int(q.iloc[0]["round"])


def first_round_above(df: pd.DataFrame, metric: str, threshold: float) -> Optional[int]:
    q = df[df[metric] > threshold]
    if q.empty:
        return None
    return int(q.iloc[0]["round"])


def expected_rounds_from_env(env: Dict[str, str], fallback: int) -> int:
    try:
        return int(env.get("ROUNDS", str(fallback)))
    except ValueError:
        return fallback


def safe_mean(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    if df.empty:
        return pd.Series({c: float("nan") for c in cols})
    return df[cols].mean()


def build_report(results_dir: str, env_path: Optional[str] = None) -> str:
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    env = read_env_file(env_path) if env_path else {}

    train_files = sorted(glob.glob(os.path.join(results_dir, "*_train_metrics.csv")))
    final_files = sorted(glob.glob(os.path.join(results_dir, "*_final.csv")))
    agg_files = sorted(glob.glob(os.path.join(results_dir, "*_agg_metrics.csv")))
    pretrain_path = os.path.join(results_dir, "pretrain_summary.csv")
    local_summary_path = os.path.join(results_dir, "local_last_summary.csv")
    plots_dir = os.path.join(results_dir, "plots")
    plot_files = sorted(glob.glob(os.path.join(plots_dir, "*.png")))

    lines: List[str] = []
    lines.append("# TinyFed Detailed Results Report")
    lines.append("")
    lines.append("## 1. Run Context")
    lines.append(f"- Generated at: `{timestamp}`")
    lines.append(f"- Results directory: `{results_dir}`")
    lines.append(f"- `.env` source: `{env_path or 'n/a'}`")
    if env:
        keys = [
            "ROUNDS",
            "EPOCHS_PER_ROUND",
            "BATCH_SIZE",
            "LEARNING_RATE",
            "TRAIN_SIZE",
            "VAL_SIZE",
            "ANOMALY_FRAC",
        ]
        cfg_rows = [[k, env.get(k, "n/a")] for k in keys]
        lines.append("")
        lines.append("Configuration snapshot:")
        lines.append("")
        lines.append(markdown_table(["Parameter", "Value"], cfg_rows))
    lines.append("")
    lines.append("Detected artifacts:")
    lines.append("")
    artifact_rows = [
        ["`*_train_metrics.csv`", str(len(train_files))],
        ["`*_final.csv`", str(len(final_files))],
        ["`*_agg_metrics.csv`", str(len(agg_files))],
        ["`pretrain_summary.csv`", "yes" if os.path.exists(pretrain_path) else "no"],
        ["`local_last_summary.csv`", "yes" if os.path.exists(local_summary_path) else "no"],
        ["`plots/*.png`", str(len(plot_files))],
    ]
    lines.append(markdown_table(["Artifact", "Count/Presence"], artifact_rows))

    if not train_files:
        lines.append("")
        lines.append("## 2. Training-Round Analysis")
        lines.append("No `*_train_metrics.csv` files found. Cannot analyze per-round behavior.")
        return "\n".join(lines) + "\n"

    # Load per-client train files
    per_client: Dict[str, pd.DataFrame] = {}
    for p in train_files:
        cid = os.path.basename(p).replace("_train_metrics.csv", "")
        df = pd.read_csv(p).sort_values("round").reset_index(drop=True)
        df["client"] = cid
        per_client[cid] = df

    all_rounds = pd.concat(list(per_client.values()), ignore_index=True)
    metric_cols = ["acc", "recall", "f1", "mse", "mae", "rss_mb", "time_per_sample_ms"]
    by_round = all_rounds.groupby("round", as_index=False)[metric_cols].mean()
    expected_rounds = expected_rounds_from_env(env, fallback=int(by_round["round"].max() + 1))

    lines.append("")
    lines.append("## 2. Global Training Dynamics (mean across clients)")
    first = by_round.iloc[0]
    last = by_round.iloc[-1]
    dyn_rows = [
        ["First round", str(int(first["round"]))],
        ["Last round", str(int(last["round"]))],
        ["Mean ACC (first -> last)", f"{f6(first['acc'])} -> {f6(last['acc'])}"],
        ["Mean Recall (first -> last)", f"{f6(first['recall'])} -> {f6(last['recall'])}"],
        ["Mean F1 (first -> last)", f"{f6(first['f1'])} -> {f6(last['f1'])}"],
        ["Mean MSE (first -> last)", f"{f6(first['mse'])} -> {f6(last['mse'])}"],
        ["Mean MAE (first -> last)", f"{f6(first['mae'])} -> {f6(last['mae'])}"],
        ["Mean RSS MB (first -> last)", f"{f6(first['rss_mb'])} -> {f6(last['rss_mb'])}"],
        [
            "Mean time/sample ms (first -> last)",
            f"{f6(first['time_per_sample_ms'])} -> {f6(last['time_per_sample_ms'])}",
        ],
    ]
    lines.append(markdown_table(["Signal", "Value"], dyn_rows))

    r_acc_90 = first_round_at_or_above(by_round, "acc", 0.90)
    r_rec_80 = first_round_at_or_above(by_round, "recall", 0.80)
    r_f1_85 = first_round_at_or_above(by_round, "f1", 0.85)
    lines.append("")
    lines.append("Convergence checkpoints:")
    lines.append("")
    cp_rows = [
        ["First round with mean ACC >= 0.90", str(r_acc_90) if r_acc_90 is not None else "not reached"],
        ["First round with mean Recall >= 0.80", str(r_rec_80) if r_rec_80 is not None else "not reached"],
        ["First round with mean F1 >= 0.85", str(r_f1_85) if r_f1_85 is not None else "not reached"],
    ]
    lines.append(markdown_table(["Checkpoint", "Round"], cp_rows))

    tail_window = min(5, len(by_round))
    tail = by_round.tail(tail_window)
    lines.append("")
    lines.append(f"Last-{tail_window}-round stability (global mean):")
    lines.append("")
    stab_cols = ["acc", "recall", "f1", "mse", "mae", "time_per_sample_ms"]
    stab_rows = []
    for c in stab_cols:
        stab_rows.append([c, f6(tail[c].mean()), f6(tail[c].std(ddof=0))])
    lines.append(markdown_table(["Metric", "Mean", "Std"], stab_rows))

    lines.append("")
    lines.append("## 3. Per-Client Analysis")
    client_rows = []
    for cid, df in sorted(per_client.items()):
        d0 = df.iloc[0]
        dN = df.iloc[-1]
        r_rec_pos = first_round_above(df, "recall", 0.0)
        r_f1_pos = first_round_above(df, "f1", 0.0)
        client_rows.append(
            [
                cid,
                str(len(df)),
                f6(d0["acc"]),
                f6(dN["acc"]),
                f6(dN["acc"] - d0["acc"]),
                f6(dN["recall"] - d0["recall"]),
                f6(dN["f1"] - d0["f1"]),
                f6(dN["mse"] - d0["mse"]),
                f6(dN["mae"] - d0["mae"]),
                str(r_rec_pos) if r_rec_pos is not None else "never",
                str(r_f1_pos) if r_f1_pos is not None else "never",
                f6(df["rss_mb"].min()),
                f6(df["rss_mb"].max()),
                f6(df["time_per_sample_ms"].mean()),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Client",
                "Rounds",
                "ACC@R0",
                "ACC@Rend",
                "ACC delta",
                "Recall delta",
                "F1 delta",
                "MSE delta",
                "MAE delta",
                "First Recall>0",
                "First F1>0",
                "RSS min",
                "RSS max",
                "Mean ms/sample",
            ],
            client_rows,
        )
    )

    lines.append("")
    lines.append("Best round per client (by F1):")
    lines.append("")
    best_rows = []
    for cid, df in sorted(per_client.items()):
        b = df.loc[df["f1"].idxmax()]
        best_rows.append(
            [
                cid,
                str(int(b["round"])),
                f6(b["acc"]),
                f6(b["recall"]),
                f6(b["f1"]),
                f6(b["mse"]),
                f6(b["mae"]),
            ]
        )
    lines.append(markdown_table(["Client", "Best round", "ACC", "Recall", "F1", "MSE", "MAE"], best_rows))

    # Final local_last summary from *_final.csv
    lines.append("")
    lines.append("## 4. Final Validation (`*_final.csv`)")
    final_rows_raw = []
    for p in final_files:
        cid = os.path.basename(p).replace("_final.csv", "")
        df = pd.read_csv(p)
        if "model" not in df.columns:
            continue
        local = df[df["model"] == "local_last"]
        if local.empty:
            continue
        r = local.iloc[0]
        final_rows_raw.append({"client": cid, "acc": float(r["acc"]), "recall": float(r["recall"]), "f1": float(r["f1"])})

    if final_rows_raw:
        final_df = pd.DataFrame(final_rows_raw).sort_values("client").reset_index(drop=True)
        lines.append("Per-client final local_last metrics:")
        lines.append("")
        rows = [
            [row["client"], f6(row["acc"]), f6(row["recall"]), f6(row["f1"])]
            for _, row in final_df.iterrows()
        ]
        lines.append(markdown_table(["Client", "ACC", "Recall", "F1"], rows))

        fm = final_df[["acc", "recall", "f1"]].agg(["mean", "std", "min", "max"])
        lines.append("")
        lines.append("Distribution across clients:")
        lines.append("")
        dist_rows = []
        for metric in ["acc", "recall", "f1"]:
            dist_rows.append([metric, f6(fm.loc["mean", metric]), f6(fm.loc["std", metric]), f6(fm.loc["min", metric]), f6(fm.loc["max", metric])])
        lines.append(markdown_table(["Metric", "Mean", "Std", "Min", "Max"], dist_rows))
    else:
        final_df = pd.DataFrame(columns=["client", "acc", "recall", "f1"])
        lines.append("No usable local_last rows found in `*_final.csv`.")

    # Pretrain comparison
    lines.append("")
    lines.append("## 5. Pretrain vs Final")
    if os.path.exists(pretrain_path):
        pre = pd.read_csv(pretrain_path)
        required = {"client", "acc", "recall", "f1"}
        if required.issubset(pre.columns):
            pre = pre[["client", "acc", "recall", "f1"]].copy()
            pre[["acc", "recall", "f1"]] = pre[["acc", "recall", "f1"]].astype(float)
            pmean = safe_mean(pre, ["acc", "recall", "f1"])
            lines.append("Pretrain mean metrics:")
            lines.append("")
            lines.append(markdown_table(["Metric", "Mean"], [[m, f6(pmean[m])] for m in ["acc", "recall", "f1"]]))

            if not final_df.empty:
                fmean = safe_mean(final_df, ["acc", "recall", "f1"])
                delta = fmean - pmean
                lines.append("")
                lines.append("Final local_last mean and delta vs pretrain:")
                lines.append("")
                delta_rows = []
                for m in ["acc", "recall", "f1"]:
                    delta_rows.append([m, f6(fmean[m]), f6(delta[m])])
                lines.append(markdown_table(["Metric", "Final mean", "Delta vs pretrain"], delta_rows))

                merged = pre.merge(final_df, on="client", suffixes=("_pretrain", "_final"))
                if not merged.empty:
                    lines.append("")
                    lines.append("Per-client delta (final - pretrain):")
                    lines.append("")
                    merge_rows = []
                    for _, r in merged.sort_values("client").iterrows():
                        merge_rows.append(
                            [
                                r["client"],
                                f6(r["acc_final"] - r["acc_pretrain"]),
                                f6(r["recall_final"] - r["recall_pretrain"]),
                                f6(r["f1_final"] - r["f1_pretrain"]),
                            ]
                        )
                    lines.append(markdown_table(["Client", "ACC delta", "Recall delta", "F1 delta"], merge_rows))
        else:
            lines.append("`pretrain_summary.csv` found, but missing required columns: `client, acc, recall, f1`.")
    else:
        lines.append("No `pretrain_summary.csv` file found.")

    # Aggregator consistency
    lines.append("")
    lines.append("## 6. Aggregator Consistency (`*_agg_metrics.csv`)")
    if agg_files:
        agg = pd.read_csv(agg_files[0]).sort_values("round").reset_index(drop=True)
        agg_rows = []
        agg_rows.append(["File used", os.path.basename(agg_files[0])])
        agg_rows.append(["Rows", str(len(agg))])
        agg_rows.append(["Round min", str(int(agg["round"].min()))])
        agg_rows.append(["Round max", str(int(agg["round"].max()))])
        agg_rows.append(["Expected rounds (from env/by data)", str(expected_rounds)])
        if "collected" in agg.columns:
            agg_rows.append(["Unique collected counts", ", ".join(map(str, sorted(agg["collected"].dropna().unique().tolist())))])
        if "published" in agg.columns:
            agg_rows.append(["Unique published flags", ", ".join(map(str, sorted(agg["published"].dropna().unique().tolist())))])
        lines.append(markdown_table(["Check", "Value"], agg_rows))

        # Compare aggregator means with means computed from client files
        compare_cols = [("mean_acc", "acc"), ("mean_rec", "recall"), ("mean_f1", "f1")]
        dif_rows = []
        merged = agg.merge(by_round[["round", "acc", "recall", "f1"]], on="round", how="inner")
        for agg_c, train_c in compare_cols:
            if agg_c not in merged.columns:
                continue
            diff = (merged[agg_c] - merged[train_c]).abs()
            dif_rows.append([f"{agg_c} vs {train_c}", f6(diff.mean()), f6(diff.max())])
        if dif_rows:
            lines.append("")
            lines.append("Aggregator-vs-client-mean absolute difference:")
            lines.append("")
            lines.append(markdown_table(["Comparison", "Mean abs diff", "Max abs diff"], dif_rows))
    else:
        lines.append("No `*_agg_metrics.csv` file found.")

    # Plot inventory
    lines.append("")
    lines.append("## 7. Plot Inventory")
    if plot_files:
        families: Dict[str, int] = {}
        for p in plot_files:
            name = os.path.basename(p)
            family = name.split("_")[0]
            families[family] = families.get(family, 0) + 1
        fam_rows = [[k, str(v)] for k, v in sorted(families.items())]
        lines.append(markdown_table(["Plot family prefix", "Count"], fam_rows))
        lines.append("")
        lines.append("Files:")
        for p in plot_files:
            lines.append(f"- `{os.path.relpath(p, results_dir)}`")
    else:
        lines.append("No PNG plots found in `plots/`.")

    # Observations
    lines.append("")
    lines.append("## 8. Automated Observations")
    obs: List[str] = []
    if r_acc_90 is not None and r_acc_90 <= max(1, expected_rounds // 3):
        obs.append(f"Model reaches mean ACC >= 0.90 early (round {r_acc_90}).")
    elif r_acc_90 is not None:
        obs.append(f"Model reaches mean ACC >= 0.90 at round {r_acc_90}.")
    else:
        obs.append("Mean ACC never reached 0.90 in available rounds.")

    if tail["acc"].std(ddof=0) < 0.01 and tail["f1"].std(ddof=0) < 0.01:
        obs.append("Last rounds show stable ACC/F1 (low variance).")
    else:
        obs.append("Last rounds still show noticeable ACC/F1 variance.")

    if first["recall"] == 0 and last["recall"] > 0.5:
        obs.append("Recall starts at 0 and later recovers strongly, indicating delayed minority-class learning.")

    rss_span_global = all_rounds["rss_mb"].max() - all_rounds["rss_mb"].min()
    if rss_span_global < 2.0:
        obs.append(f"RSS memory is stable across run (global span {rss_span_global:.3f} MB).")
    else:
        obs.append(f"RSS memory varies notably (global span {rss_span_global:.3f} MB).")

    for o in obs:
        lines.append(f"- {o}")

    lines.append("")
    lines.append("## 9. Repro Commands")
    lines.append("```bash")
    lines.append("docker compose up --build")
    lines.append("python3 analyze.py")
    lines.append("python3 report_results.py --results-dir results --output results/detailed_report.md")
    lines.append("```")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a detailed markdown report from TinyFed results CSV artifacts."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory (default: results).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env with run configuration (default: .env).",
    )
    parser.add_argument(
        "--output",
        default="results/detailed_report.md",
        help="Output markdown path. Use '-' to print to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.results_dir, args.env_file)

    if args.output == "-":
        print(report, end="")
        return 0

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[REPORT] Detailed report saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
