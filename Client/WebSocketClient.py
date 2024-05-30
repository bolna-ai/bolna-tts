import websocket
import json
import time
conn = websocket.create_connection("ws://127.0.0.1:4680/ws/tts")

for i in range(10):
    conn.send(json.dumps({
        "text": "hello world",
        "speacker_id": "EN-US",
        "sdp_ratio": 0.2,
        "noise_scale": 0.6,
        "noise_scale_w": 0.8,
        "speed": 1.0,
        "model":"DummyTTS",
        "sample_rate":8000
    }))
    time.sleep(1)


time.sleep(5)