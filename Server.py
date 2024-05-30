from fastapi import FastAPI, status, HTTPException, Request, WebSocket
from pydantic import BaseModel
import sys
from typing import List, Union
from TTS import TTSOrchestrator, DummyTTS_SYNC, MeloTTS_SYNC, StyleTTS_SYNC
from contextlib import asynccontextmanager
import uvicorn
import torch
from TTS.Modules.base import TTS
from WebScoketThread import WebsocketClient

TTS_Orchestrator:TTSOrchestrator = None
WebSocketClients:List[WebsocketClient] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global TTS_Orchestrator
    TTS_Orchestrator = TTSOrchestrator()
    yield
    del TTS_Orchestrator
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)
class TTS_HTTP(BaseModel):
    model:str
    config:Union[DummyTTS_SYNC.Config,MeloTTS_SYNC.Config,StyleTTS_SYNC.Config]

@app.websocket("/ws/tts")
async def tts_ws(websocket:WebSocket):
    await websocket.accept()
    client = WebsocketClient(TTS_Orchestrator=TTS_Orchestrator,websocket=websocket)
    client.start()
    WebSocketClients.append(client)
    

@app.post("/tts")
def tts_http(response:TTS_HTTP):
    model:TTS = TTS_Orchestrator.getModel(response.model)
    if model == None: raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'model {response.model} is not present in model Orchestrator available model',
            )
    print(response)
    # config = TTS_Orchestrator.getModelConfig(response.model).parse_obj(response.model_config)
    return next(model.systhesized(response.config))
    # model
if __name__ == "__main__":
    uvicorn.run("Server:app",host='0.0.0.0',port=4680,reload=True)