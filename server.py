from fastapi import FastAPI, HTTPException, WebSocket, Response
from pydantic import BaseModel
import torch
import numpy as np
import io
import scipy
import uuid
import base64
import json
import asyncio
import time
import websockets
import logging
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


tts_coqui, tts_matcha = None, None

try:
    from matcha_tts import MatchaTTSWrapper
    model_name = "matcha_ljspeech_finetuned" # , matcha_lj    speech, matcha_vctk
    vocoder_name = "hifigan_univ_v1" # hifigan_T2_v1 , hifigan_univ_v1, vocos

    tts_matcha = MatchaTTSWrapper(
    model_name=model_name, 
    vocoder_name=vocoder_name,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    model_path=f"/home/ubuntu/models/checkpoint_epoch_4599.ckpt",
    vocoder_path=f"/home/ubuntu/.local/share/matcha_tts/{vocoder_name}",
    speaking_rate=1.0
    )
except Exception as e:
    logger.info("Running in the coqui environment. No matcha present.")

try:
    from coqui_tts import CoquiTTSWrapper
    tts_coqui = CoquiTTSWrapper()
    logger.info("loaded xtts wrapper")
except Exception as e:
    logger.info(f"Running in the matcha environment {e}")

app = FastAPI()

class TextPayload(BaseModel):
    text: str
    model: str
    voice: str
    language: Optional[str] = "en"

coqui_voice_list = os.getenv("COQUI_VOICE_LIST").split(",") 
currently_processing = False

@app.websocket("/generate")
async def websocket_endpoint(websocket: WebSocket):
    global currently_processing

    await websocket.accept()
    input_queue = asyncio.Queue()
    async def listen_for_text():
        try:
            while True:
                request = await websocket.receive_json()
                logger.info(f"Received: {request}")
                if request["model"] == "xtts":
                    input_queue.put_nowait(request)
                    logger.info("Added to queue")
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed with code {e.code}")

    async def generate_audio():
        global currently_processing
        try:
            while True:
                logger.info("Getting the next message from the queue")
                request = await input_queue.get()
                start_time = time.time()
                i = 0 
                logger.info(f"Starting new stream for {request}")
                for chunk in tts_coqui.generate_stream(request["text"], request["language"], request["voice"]):
                    logger.info(f"Chunk {i} generation time {time.time() - start_time}")
                    await websocket.send_bytes(chunk)
                    logger.info(f"Chunk {i} Send time {time.time() - start_time}")
                    i+=1
                if "end_of_stream" in request and request["end_of_stream"]:
                    logger.info("entire text stream has audio generated, sending end of stream signal")
                    await websocket.send_bytes(b'\x00')
                currently_processing = False
                logger.info(f"Finished processing request with {request}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed with code {e.code}")

    try:
        await asyncio.gather(listen_for_text(), generate_audio())
    except Exception as e:
        logger.info(f"An error occurred: {e}")

    logger.info("WebSocket connection handler completed.")


@app.post("/generate", response_class=Response)
async def generate_audio(payload: TextPayload):
    try:
        request_id = str(uuid.uuid4())
        if payload.model == "matcha":
            audio_bytes = await tts_matcha.generate(payload.text, request_id)
        else:
            logger.info("Generating audio")
            audio_bytes = tts_coqui.generate(payload.text, payload.voice)
        
        return Response(audio_bytes, media_type="audio/wav")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
