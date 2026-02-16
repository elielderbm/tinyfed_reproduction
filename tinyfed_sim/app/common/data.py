import numpy as np
import pandas as pd
import random

# -------- Dataset fixo para pré-treino --------
def make_synthetic_dataset(train_size: int, val_size: int, anomaly_frac: float, seed: int, client_id: str = None):
    rng = np.random.default_rng(seed)

    # client_id influencia a distribuição (não-IID)
    cid_hash = sum(bytearray(client_id.encode())) if client_id else seed
    mode = cid_hash % 3

    norm_means = [25.0, 50.0, 300.0, 3.3]
    ano_means  = [30.0, 35.0, 380.0, 3.4]
    norm_stds  = [3.0, 8.0, 60.0, 0.05]
    ano_stds   = [4.0, 8.0, 80.0, 0.07]

    frac = anomaly_frac
    if mode == 0: frac = 0.2
    elif mode == 1: frac = 0.8
    elif mode == 2: frac = 0.5

    def _gen(n, means, stds, label):
        x = np.stack([rng.normal(m,s,n) for m,s in zip(means,stds)], axis=1)
        y = np.full((n,1),label)
        return x,y

    n_anom_tr = int(train_size*frac); n_norm_tr = train_size-n_anom_tr
    n_anom_v  = int(val_size*frac);  n_norm_v  = val_size-n_anom_v

    x_tr = np.vstack([_gen(n_norm_tr,norm_means,norm_stds,0)[0],
                      _gen(n_anom_tr, ano_means, ano_stds,1)[0]])
    y_tr = np.vstack([np.zeros((n_norm_tr,1)), np.ones((n_anom_tr,1))])

    x_v = np.vstack([_gen(n_norm_v,norm_means,norm_stds,0)[0],
                     _gen(n_anom_v, ano_means, ano_stds,1)[0]])
    y_v = np.vstack([np.zeros((n_norm_v,1)), np.ones((n_anom_v,1))])

    mu = x_tr.mean(axis=0,keepdims=True)
    sigma = x_tr.std(axis=0,keepdims=True)+1e-8
    x_tr = (x_tr-mu)/sigma
    x_v  = (x_v -mu)/sigma

    df_tr = pd.DataFrame(x_tr,columns=["temp","hum","lux","volt"]); df_tr["label"]=y_tr
    df_v  = pd.DataFrame(x_v, columns=["temp","hum","lux","volt"]); df_v["label"]=y_v

    return df_tr, df_v

# -------- Geração contínua para tempo real --------
def generate_sensor_sample(client_id=None):
    rng = np.random.default_rng(sum(bytearray(client_id.encode())) if client_id else None)

    temp = rng.normal(25, 3)
    hum  = rng.normal(50, 8)
    lux  = rng.normal(300, 60)
    volt = rng.normal(3.3, 0.05)
    label = 0

    anomaly_bias = (sum(bytearray(client_id.encode())) % 30) / 100.0
    if random.random() < (0.15 + anomaly_bias):
        temp = rng.normal(32, 3)
        hum  = rng.normal(20, 6)
        lux  = rng.normal(450, 80)
        volt = rng.normal(3.6, 0.05)
        label = 1

    return [temp, hum, lux, volt], label
