import os, time, csv, numpy as np, psutil
from common.data import make_synthetic_dataset
from common.fl_model import MLPBinary4_16_8_4_2
from common.metrics import classification_metrics, regression_errors
from common.mqtt_utils import (
    make_client, publish_json, subscribe,
    loop_start, loop_stop, TOPIC_LOCAL_WEIGHTS, TOPIC_GLOBAL_WEIGHTS
)

# ===== Config =====
BROKER_HOST = os.getenv("BROKER_HOST", "mosquitto")
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))

ROUNDS = int(os.getenv("ROUNDS", "25"))
EPOCHS_PER_ROUND = int(os.getenv("EPOCHS_PER_ROUND", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LR = float(os.getenv("LEARNING_RATE", "0.05"))

NUM_FEATURES = int(os.getenv("NUM_FEATURES", "4"))
H1 = int(os.getenv("HIDDEN_1", "16"))
H2 = int(os.getenv("HIDDEN_2", "8"))
H3 = int(os.getenv("HIDDEN_3", "4"))
OUT = int(os.getenv("NUM_CLASSES", "2"))

TRAIN_SIZE = int(os.getenv("TRAIN_SIZE", "1400"))
VAL_SIZE = int(os.getenv("VAL_SIZE", "600"))
ANOMALY_FRAC = float(os.getenv("ANOMALY_FRAC", "0.15"))

FINAL_WAIT_SEC = float(os.getenv("FINAL_WAIT_SEC", "15.0"))  # espera extra após última rodada
WAIT_PER_ROUND = float(os.getenv("WAIT_PER_ROUND", "10.0"))  # espera por rodada específica

host = os.uname().nodename
seed = abs(hash(host)) % (2**31 - 1)
print(f"[CLIENT {host}] Broker={BROKER_HOST}:{BROKER_PORT} | Rounds={ROUNDS} | seed={seed}", flush=True)

# ===== Dados =====
df_tr, df_v = make_synthetic_dataset(TRAIN_SIZE, VAL_SIZE, ANOMALY_FRAC, seed=seed)
X_tr = df_tr[["temp","hum","lux","volt"]].values.astype(float)
y_tr = df_tr["label"].values.astype(int).reshape(-1,1)
X_v  = df_v[["temp","hum","lux","volt"]].values.astype(float)
y_v  = df_v["label"].values.astype(int).reshape(-1,1)

# ===== Modelo =====
model = MLPBinary4_16_8_4_2(
    input_dim=NUM_FEATURES, h1=H1, h2=H2, h3=H3,
    out_dim=OUT, lr=LR, seed=seed % 10000
)

# ===== Pré-treino =====
from common.metrics import classification_metrics as cm
y_pred_cls, _ = model.predict(X_v)
acc, rec, f1 = cm(y_v, y_pred_cls)

os.makedirs("/results", exist_ok=True)
pretrain_csv = "/results/pretrain_summary.csv"
file_exists = os.path.exists(pretrain_csv)
with open(pretrain_csv, "a", newline="") as f:
    w = csv.writer(f)
    if not file_exists:
        w.writerow(["client","acc","recall","f1"])
    w.writerow([host, f"{acc:.6f}", f"{rec:.6f}", f"{f1:.6f}"])

print(f"[CLIENT {host}] Pré-treino salvo em {pretrain_csv}", flush=True)

# ===== MQTT =====
client = make_client(f"client_{host}")
loop_start(client)

# Armazena pesos globais por rodada para garantir aplicação correta
pending_global_by_round = {}

def on_global_weights(payload):
    if "weights" in payload and "round" in payload:
        r = int(payload["round"])
        pending_global_by_round[r] = payload["weights"]

subscribe(client, TOPIC_GLOBAL_WEIGHTS, on_global_weights)

def batches(X, y, bs):
    n = X.shape[0]
    idx = np.arange(n)
    for i in range(0, n, bs):
        sel = idx[i:i+bs]
        yield X[sel], y[sel]

# ===== CSV =====
round_csv = f"/results/{host}_train_metrics.csv"
final_csv = f"/results/{host}_final.csv"
with open(round_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round","acc","recall","f1","mse","mae","rss_mb","time_per_sample_ms","timestamp_sec"])

# ===== Snapshots =====
proc = psutil.Process()
local_last_weights = None

# ===== Loop de rounds =====
for r in range(ROUNDS):
    t0 = time.time()
    for _ in range(EPOCHS_PER_ROUND):
        idx = np.arange(X_tr.shape[0])
        np.random.shuffle(idx)
        X_tr_s = X_tr[idx]; y_tr_s = y_tr[idx]
        for i in range(0, X_tr_s.shape[0], BATCH_SIZE):
            Xb = X_tr_s[i:i+BATCH_SIZE]
            yb = y_tr_s[i:i+BATCH_SIZE]
            yb_oh = model.one_hot(yb, num_classes=OUT)
            loss, grads, _ = model.loss_and_grads(Xb, yb_oh)
            model.step(grads)

    y_pred_cls, y_pred_prob = model.predict(X_tr)
    acc, rec, f1 = classification_metrics(y_tr, y_pred_cls)
    mse, mae = regression_errors(y_tr, y_pred_prob)

    rss_mb = proc.memory_info().rss / (1024*1024)
    elapsed = time.time() - t0
    t_per_sample = 1000.0 * elapsed / X_tr.shape[0]

    print(f"[CLIENT {host}] round={r} | acc={acc:.4f} rec={rec:.4f} f1={f1:.4f} mse={mse:.4f} mae={mae:.4f}", flush=True)

    with open(round_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([r, f"{acc:.6f}", f"{rec:.6f}", f"{f1:.6f}",
                    f"{mse:.6f}", f"{mae:.6f}", f"{rss_mb:.3f}",
                    f"{t_per_sample:.6f}", int(time.time())])

    # publica pesos + métricas
    publish_json(client, TOPIC_LOCAL_WEIGHTS, {
        "round": r,
        "weights": model.get_weights(),
        "num_samples": int(X_tr.shape[0]),
        "accuracy": float(acc),
        "recall": float(rec),
        "f1": float(f1)
    })

    # guarda o local ANTES do global na última rodada
    if r == ROUNDS - 1:
        local_last_weights = model.get_weights()

    # aguarda e aplica APENAS o global da rodada r
    t_wait = time.time()
    while (r not in pending_global_by_round) and (time.time() - t_wait < WAIT_PER_ROUND):
        time.sleep(0.05)

    if r in pending_global_by_round:
        model.set_weights(pending_global_by_round[r])
        del pending_global_by_round[r]
        print(f"[CLIENT {host}] round={r} | global weights applied", flush=True)
    else:
        print(f"[CLIENT {host}] round={r} | global NOT received (timeout={WAIT_PER_ROUND:.1f}s)", flush=True)

# ===== Espera extra para global final (se chegou atrasado) =====
t0 = time.time()
while ((ROUNDS-1) not in pending_global_by_round) and (time.time() - t0 < FINAL_WAIT_SEC):
    time.sleep(0.05)

if (ROUNDS-1) in pending_global_by_round:
    model.set_weights(pending_global_by_round[ROUNDS-1])
    del pending_global_by_round[ROUNDS-1]
    print(f"[CLIENT {host}] última rodada | global weights applied (late)", flush=True)

# ===== Validação final: apenas local_last =====
if local_last_weights is None:
    local_last_weights = model.get_weights()

model_local_last = MLPBinary4_16_8_4_2(
    input_dim=NUM_FEATURES, h1=H1, h2=H2, h3=H3,
    out_dim=OUT, lr=LR, seed=seed % 10000
)
model_local_last.set_weights(local_last_weights)

loc_pred_cls, _ = model_local_last.predict(X_v)
loc_acc, loc_rec, loc_f1 = cm(y_v, loc_pred_cls)

with open(final_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model","acc","recall","f1"])
    w.writerow(["local_last", f"{loc_acc:.6f}", f"{loc_rec:.6f}", f"{loc_f1:.6f}"])

print(f"[CLIENT {host}] === Validation: Local_last ===", flush=True)
print(f"[CLIENT {host}] Local_last -> acc={loc_acc:.4f} rec={loc_rec:.4f} f1={loc_f1:.4f}", flush=True)

loop_stop(client)
