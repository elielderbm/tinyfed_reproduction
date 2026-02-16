import json, os
import paho.mqtt.client as mqtt

BROKER_HOST = os.getenv("BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))

TOPIC_LOCAL_WEIGHTS = "tinyfed/weights/local"
TOPIC_GLOBAL_WEIGHTS = "tinyfed/weights/global"

def make_client(client_id: str) -> mqtt.Client:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=True)
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    return client

def publish_json(client: mqtt.Client, topic: str, payload: dict):
    data = json.dumps(payload)
    client.publish(topic, data, qos=1)

def subscribe(client: mqtt.Client, topic: str, on_message):
    def _on_message(_client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            payload = {}
        on_message(payload)
    client.subscribe(topic, qos=1)
    client.on_message = _on_message

def loop_start(client: mqtt.Client):
    client.loop_start()

def loop_stop(client: mqtt.Client):
    client.loop_stop()
