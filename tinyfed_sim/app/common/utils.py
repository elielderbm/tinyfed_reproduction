import json
import numpy as np

def publish_json(client, topic, payload):
    """Publica um dicionário como JSON no broker MQTT"""
    client.publish(topic, json.dumps(payload))

def cm(y_true, y_pred):
    """
    Calcula Accuracy, Recall e F1-score binário (anomalia=1).
    Retorna: acc, recall, f1
    """
    tp = int(np.sum((y_true==1) & (y_pred==1)))
    tn = int(np.sum((y_true==0) & (y_pred==0)))
    fp = int(np.sum((y_true==0) & (y_pred==1)))
    fn = int(np.sum((y_true==1) & (y_pred==0)))

    acc = (tp+tn) / max(len(y_true),1)
    recall = tp / max((tp+fn),1)
    precision = tp / max((tp+fp),1)
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0

    return acc, recall, f1

def batches(X, y, batch_size):
    """Gera lotes (Xb,yb) a partir de arrays X,y"""
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
