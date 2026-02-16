# TinyFed – Simulação + Análises (Docker + Python + MQTT)

Este projeto implementa **a reprodução fiel da metodologia do artigo _TinyFed: Lightweight Federated Learning for Constrained IoT Devices_** usando **Docker + Python + MQTT**.  
Inclui **código de simulação**, **coleta de métricas**, e **scripts de análise** para gerar gráficos e tabelas equivalentes às apresentadas no artigo.

---

## 🎯 Objetivo
Permitir que você rode **todo o ciclo federado (TinyFed)** em ambiente controlado, coletando e analisando:
- **Métricas de erro**: MSE e MAE.
- **Métricas de classificação**: Accuracy, Recall e F1-score.
- **Comparação Local vs Agregado** (validando a eficácia do FedAvg).
- **Consumo de memória (RSS do processo)**, aproximando a análise de HEAP feita no ESP32.
- **Tempo por amostra**, comparável aos valores de referência reportados no artigo.

---

## 🗂 Estrutura do Projeto

```
tinyfed-sim-analysis/
├─ docker-compose.yml       # Orquestra broker, agregador e clientes
├─ .env                     # Configuração de parâmetros de treino
├─ README.md                # Este guia detalhado
├─ analyze.py               # Script para processar resultados e gerar gráficos/tabelas
├─ results/                 # (criado em runtime) CSVs e plots
├─ mosquitto/
│  └─ mosquitto.conf        # Configuração do broker MQTT
└─ app/
   ├─ requirements.txt      # Dependências Python
   ├─ Dockerfile            # Imagem base para aggregator/client
   ├─ entrypoint.sh         # Script de entrada que decide o papel (client/aggregator)
   ├─ aggregator.py         # Implementa FedAvg e coordenação global
   ├─ client.py             # Simula ESP32: coleta dados, treina, envia pesos
   └─ common/
      ├─ mqtt_utils.py      # Utilitários MQTT (publicar/assinar)
      ├─ data.py            # Geração de dataset sintético (temp, hum, lux, volt)
      ├─ metrics.py         # Cálculo de métricas (acc, recall, f1, mse, mae)
      └─ fl_model.py        # Implementação da MLP 4-16-8-4-2 com sigmoid
```

---

## ⚙️ Configuração

### Pré-requisitos
- **Docker** e **Docker Compose** instalados.
- Portas **1883** (MQTT) e **9001** (WebSocket) livres.

### Parâmetros (.env)
Edite `.env` para ajustar:
```
ROUNDS=25           # Número de rodadas de FL
EPOCHS_PER_ROUND=1  # Épocas locais por rodada
BATCH_SIZE=64       # Tamanho do batch
LEARNING_RATE=0.05  # Taxa de aprendizado
TRAIN_SIZE=1400     # Amostras de treino por cliente
VAL_SIZE=600        # Amostras de validação
ANOMALY_FRAC=0.15   # % de anomalias simuladas
```

---

## ▶️ Execução

1. **Subir o ambiente federado**
```bash
docker compose up --build
```
- Sobe o broker MQTT, o agregador e **3 clientes simulando ESP32**.
- Cada cliente:
  - Gera dataset sintético local (com normalidade e anomalias).
  - Treina a rede neural **MLP 4-16-8-4-2 com sigmoid**.
  - Envia pesos para o agregador via MQTT.
  - Recebe os pesos globais atualizados (FedAvg).

2. **(Opcional) Escalar clientes**
```bash
docker compose up --build --scale client=5
```

3. **Ver métricas em tempo real**
- Cada cliente imprime no log a cada rodada:
  - Accuracy, Recall, F1
  - MSE, MAE
  - Memória RSS (MB)
  - Tempo médio por amostra (ms)

---

## 📊 Resultados e Análises

Todos os resultados ficam em `./results/`.

### Arquivos gerados por cliente
- `results/<client>_train_metrics.csv`: métricas por rodada.
- `results/<client>_final.csv`: comparação **Local vs Agregado**.

### Arquivo do agregador
- `results/aggregator_log.csv`: número de clientes por rodada.

### Script de análise
Após rodar o treinamento, execute:
```bash
python3 analyze.py
```
Gera automaticamente:
- Gráficos em `results/plots/`:
  - `mse_over_rounds.png`
  - `mae_over_rounds.png`
  - `accuracy_over_rounds.png`
  - `recall_over_rounds.png`
  - `f1_over_rounds.png`
- Tabela consolidada `results/local_vs_aggregated_summary.csv` comparando todos os clientes.

---

## 📈 Interpretação dos Resultados

- **Curvas MSE/MAE**: devem cair ao longo das rodadas, mostrando convergência.
- **Accuracy/Recall/F1**: tendem a subir, com recall variando entre clientes (heterogeneidade de dados).
- **Local vs Agregado**: o modelo agregado deve superar o modelo local, confirmando o ganho do FedAvg.
- **Memória RSS**: deve se manter estável, mostrando ausência de vazamentos (análoga ao monitoramento de HEAP do artigo).
- **Tempo por amostra**: valores próximos a 0,04s podem ser atingidos ajustando `BATCH_SIZE` e `EPOCHS_PER_ROUND`.

---

## 🔮 Extensões Futuras

- Rodar com **datasets reais** (Intel Lab, Gas Sensor Array).
- Incluir **ESP32 físico com MicroPython** para validação prática.
- Avaliar **estratégias assíncronas** e personalização por cliente.
- Estudar **segurança e privacidade** no ciclo TinyFed.

---

## ✅ Conclusão

Este ambiente entrega **uma reprodução fiel do artigo TinyFed em Docker/Python**, permitindo:
1. **Simular IoT restrito** (via containers).
2. **Rodar ciclo FL completo** (treino local + agregação global).
3. **Extrair métricas e análises idênticas às do paper**.