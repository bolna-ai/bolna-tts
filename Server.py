from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import sys
from threading import Thread
import threading
from TTS import TTSOrchestrator, MeloTTS_SYNC, StyleTTS_SYNC
from contextlib import asynccontextmanager
import uvicorn
import torch

TTS_Orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    TTS_Orchestrator = TTSOrchestrator()
    yield
    del TTS_Orchestrator
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)
class TTS_HTTP(BaseModel):
    model:str
    model_config:Union[MeloTTS_SYNC.Config,StyleTTS_SYNC.Config]


@app.post("/tts")
def tts_http(response:TTS_HTTP):
    pass
if __name__ == "__main__":
    uvicorn.run("Server:app",host='0.0.0.0',port=4680,reload=True)