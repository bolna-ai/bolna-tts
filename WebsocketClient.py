from threading import Thread
from fastapi import WebSocket
from uuid import uuid4
from TTS.Logger import  configure_logger
from typing import Union
from TTS import StyleTTS_SYNC, MeloTTS_SYNC


logger = configure_logger(__name__)
class WebsocketClient(Thread):
    def __init__(self,websocket:WebSocket,name:str=None) -> None:
        self.name = name
        if self.name == None: name = str(uuid4()) 
        self.websocket = websocket
        config_data = self.websocket.receive_json()
        logger.info(f"")
        
    def run(self) -> None:
        while True:
            conf:Union[MeloTTS_SYNC.Config,StyleTTS_SYNC.Config] = self.websocket.receive_json()


            

