import os, glob
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ====== Carregar métricas por rodada ======
round_files = glob.glob(os.path.join(RESULTS_DIR, "*_train_metrics.csv"))
if not round_files:
    print("[ANALYZE] Nenhum *_train_metrics.csv encontrado. Rode os clients antes.")
else:
    dfs = []
    for p in round_files:
        df = pd.read_csv(p)
        df["client"] = os.path.basename(p).replace("_train_metrics.csv","")
        dfs.append(df)
    all_rounds = pd.concat(dfs, ignore_index=True)

    # Média por rodada (todos os clientes)
    by_round = all_rounds.groupby("round", as_index=False)[
        ["mse","mae","acc","recall","f1","rss_mb","time_per_sample_ms"]
    ].mean()

    # ====== Função utilitária de gráfico ======
    def save_line(figpath, x, y, xlabel, ylabel, title):
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(figpath, bbox_inches="tight", dpi=140)
        plt.close()

    # Gráficos médios
    save_line(os.path.join(PLOTS_DIR,"mse_over_rounds.png"), by_round["round"], by_round["mse"], "Round", "MSE", "MSE ao longo das rodadas")
    save_line(os.path.join(PLOTS_DIR,"mae_over_rounds.png"), by_round["round"], by_round["mae"], "Round", "MAE", "MAE ao longo das rodadas")
    save_line(os.path.join(PLOTS_DIR,"accuracy_over_rounds.png"), by_round["round"], by_round["acc"], "Round", "Accuracy", "Accuracy ao longo das rodadas")
    save_line(os.path.join(PLOTS_DIR,"recall_over_rounds.png"), by_round["round"], by_round["recall"], "Round", "Recall", "Recall ao longo das rodadas")
    save_line(os.path.join(PLOTS_DIR,"f1_over_rounds.png"), by_round["round"], by_round["f1"], "Round", "F1-score", "F1 ao longo das rodadas")

    print("[ANALYZE] Gráficos de evolução salvos em results/plots/")

    # === Curvas individuais por cliente ===
    for cid, g in all_rounds.groupby("client"):
        for metric in ["mse","mae","acc","recall","f1"]:
            save_line(
                os.path.join(PLOTS_DIR, f"{metric}_over_rounds_{cid}.png"),
                g["round"], g[metric],
                "Round", metric.upper(),
                f"{metric.upper()} - Cliente {cid}"
            )

# ====== Comparação final Local_last ======
final_files = glob.glob(os.path.join(RESULTS_DIR, "*_final.csv"))
rows = []
for p in final_files:
    df = pd.read_csv(p)
    cid = os.path.basename(p).replace("_final.csv","")
    local = df[df["model"]=="local_last"].iloc[0]
    rows.append({
        "client": cid,
        "acc": local["acc"],
        "recall": local["recall"],
        "f1": local["f1"],
    })

if rows:
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "local_last_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[ANALYZE] local_last_summary.csv gerado em {summary_path}")
    print(summary)

    # Médias globais (todos os clients)
    global_mean = summary[["acc","recall","f1"]].mean()
    print("\n[ANALYZE] Desempenho médio Local_last (todos os clientes):")
    print(global_mean)

    # Gráfico de barras comparando clientes
    for _, row in summary.iterrows():
        cid = row["client"]
        metrics = ["acc","recall","f1"]
        vals = [row[m] for m in metrics]

        plt.figure()
        x = range(len(metrics))
        plt.bar(x, vals, width=0.5, label="Local_last")
        plt.xticks(x, ["Accuracy","Recall","F1"])
        plt.ylabel("Score")
        plt.title(f"Local_last - {cid}")
        plt.ylim(0,1.05)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(PLOTS_DIR,f"local_last_{cid}.png"), bbox_inches="tight", dpi=140)
        plt.close()

    # Gráfico de barras com médias globais
    metrics = ["acc","recall","f1"]
    mean_vals = [global_mean[m] for m in metrics]

    plt.figure()
    x = range(len(metrics))
    plt.bar(x, mean_vals, width=0.5, color="orange", label="Média Local_last")
    plt.xticks(x, ["Accuracy","Recall","F1"])
    plt.ylabel("Score")
    plt.title("Médias Local_last (Todos os Clients)")
    plt.ylim(0,1.05)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(PLOTS_DIR,"local_last_means.png"), bbox_inches="tight", dpi=140)
    plt.close()

    print("[ANALYZE] Gráficos de Local_last salvos em results/plots/")

else:
    print("[ANALYZE] Nenhum *_final.csv encontrado. Pulei análise final.")

# ====== Analisar pré-treino se existir ======
pretrain_path = os.path.join(RESULTS_DIR, "pretrain_summary.csv")
if os.path.exists(pretrain_path):
    pretrain = pd.read_csv(pretrain_path)
    print("\n[ANALYZE] Resumo do Pré-treino (Fase 1):")
    print(pretrain)

    if rows:
        pretrain_mean = pretrain[["acc","recall","f1"]].mean()
        print("\n[ANALYZE] Comparação médias Pré-treino vs Local_last:")
        comparison = pd.DataFrame({
            "Pretrain": pretrain_mean,
            "Local_last": global_mean
        })
        print(comparison)

        # Gráfico comparando Pré-treino vs Local_last
        plt.figure()
        metrics = ["acc","recall","f1"]
        x = range(len(metrics))
        plt.bar([i-0.2 for i in x], [pretrain_mean[m] for m in metrics], width=0.4, label="Pré-treino")
        plt.bar([i+0.2 for i in x], [global_mean[m] for m in metrics], width=0.4, label="Local_last (média)")
        plt.xticks(x, ["Accuracy","Recall","F1"])
        plt.ylabel("Score")
        plt.title("Pré-treino vs Local_last (médias)")
        plt.ylim(0,1.05)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(PLOTS_DIR,"pretrain_vs_local_last.png"), bbox_inches="tight", dpi=140)
        plt.close()
