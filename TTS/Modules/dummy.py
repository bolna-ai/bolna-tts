from .base import TTS
import torch
from base64 import b64encode
from scipy.io.wavfile import write
import time
import io
from TTS.Logger import configure_logger
from pydantic import BaseModel

logger = configure_logger(__name__)
class DummyTTS_SYNC(TTS):
    def __init__(self) -> None:
        super().__init__("DummyTTS", 24000, torch.float32)
        logger.info(f"{self.name} is loaded in cpu")
    
    class Config(BaseModel):
        text:str
        speacker_id:str
        sr:int
        sdp_ratio:float = 0.2
        noise_scale:float = 0.6
        noise_scale_w:float = 0.8
        speed:float= 1.0

    def systhesized(self,config:Config):
        self.lock.acquire()
        logger.info(f"{config}")
        __t = time.time()
        process_time = time.time() - __t
        yield {'audio': "kdkdsjf",'sr':config.sr,"time":process_time}
        self.lock.release()
        

        
