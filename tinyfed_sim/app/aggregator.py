import os, time, csv, numpy as np
from common.fl_model import MLPBinary4_16_8_4_2
from common.mqtt_utils import (
    make_client, publish_json, subscribe,
    loop_start, loop_stop, TOPIC_LOCAL_WEIGHTS, TOPIC_GLOBAL_WEIGHTS
)

# ===== Config =====
BROKER_HOST = os.getenv("BROKER_HOST", "mosquitto")
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))
ROUNDS = int(os.getenv("ROUNDS", "25"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))
WAIT_TIMEOUT = int(os.getenv("WAIT_TIMEOUT", "10"))

NUM_FEATURES = int(os.getenv("NUM_FEATURES", "4"))
H1 = int(os.getenv("HIDDEN_1", "16"))
H2 = int(os.getenv("HIDDEN_2", "8"))
H3 = int(os.getenv("HIDDEN_3", "4"))
OUT = int(os.getenv("NUM_CLASSES", "2"))

host = os.uname().nodename
print(f"[AGG] Broker={BROKER_HOST}:{BROKER_PORT} | Rounds={ROUNDS} | Clients={NUM_CLIENTS}", flush=True)

# ===== CSV =====
os.makedirs("/results", exist_ok=True)
agg_csv = f"/results/{host}_agg_metrics.csv"
with open(agg_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round","collected","published","timestamp_sec","mean_acc","mean_rec","mean_f1"])

# ===== MQTT =====
client = make_client(f"aggregator_{host}")
loop_start(client)

collected = {}

def on_local_weights(payload):
    if "round" in payload and "weights" in payload:
        r = payload["round"]
        if r not in collected:
            collected[r] = []
        collected[r].append(payload)

subscribe(client, TOPIC_LOCAL_WEIGHTS, on_local_weights)

# ===== Modelo inicial =====
model = MLPBinary4_16_8_4_2(
    input_dim=NUM_FEATURES, h1=H1, h2=H2, h3=H3,
    out_dim=OUT, lr=0.01
)

# ===== Loop =====
for r in range(ROUNDS):
    t0 = time.time()
    print(f"[AGG] round={r} esperando clientes...", flush=True)

    while (r not in collected or len(collected[r]) < NUM_CLIENTS) and (time.time()-t0 < WAIT_TIMEOUT):
        time.sleep(0.2)

    if r not in collected or len(collected[r]) == 0:
        print(f"[AGG] round={r} sem updates, pulando", flush=True)
        continue

    payloads = collected[r]
    client_weights = [p["weights"] for p in payloads]
    keys = client_weights[0].keys()

    # FedAvg simples
    avg_weights = {}
    for k in keys:
        arrs = [np.array(cw[k], dtype=np.float32) for cw in client_weights]
        avg_weights[k] = np.mean(arrs, axis=0)

    model.set_weights(avg_weights)
    publish_json(client, TOPIC_GLOBAL_WEIGHTS, {"round": r, "weights": model.get_weights()})

    # métricas médias
    accs = [p.get("accuracy", 0.0) for p in payloads]
    recs = [p.get("recall", 0.0) for p in payloads]
    f1s  = [p.get("f1", 0.0) for p in payloads]
    m_acc, m_rec, m_f1 = np.mean(accs), np.mean(recs), np.mean(f1s)

    elapsed = time.time() - t0
    print(f"[AGG] round={r} collected={len(payloads)} published | acc={m_acc:.4f} rec={m_rec:.4f} f1={m_f1:.4f} | t={elapsed:.2f}s", flush=True)

    with open(agg_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([r, len(payloads), 1, int(time.time()), f"{m_acc:.6f}", f"{m_rec:.6f}", f"{m_f1:.6f}"])

print(f"[AGG] Done all rounds.", flush=True)
loop_stop(client)
