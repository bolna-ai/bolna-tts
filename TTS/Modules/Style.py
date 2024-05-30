from .base import TTS
from .ReqModule import StyleTTS
from base64 import b64encode
from scipy.io.wavfile import write
import time
import io
from TTS.Logger import configure_logger
import torch
from pydantic import BaseModel

logger = configure_logger(__name__)
class StyleTTS_SYNC(TTS):
    def __init__(self) -> None:
        super().__init__("StyleTTS", 24000, torch.float32)
        self.model = StyleTTS()
        logger.info(f"{self.name} is loaded in {self.model.device}")

    class Config(BaseModel):
        text:str
        sr:int
        diffusion_steps:int = 5
        embedding_scale:int = 1
        

    def systhesized(self,payload:Config):
        self.lock.acquire()
        logger.info(f"StyleTTS systhesized {payload.text} and out wav of sample rate {payload.sr}")
        logger.info(f"with config: [diffusion_steps: {payload.diffusion_steps} embedding_scale: {payload.embedding_scale}]")
        __t = time.time()
        data = self.model.inference(payload.text,diffusion_steps=payload.diffusion_steps,embedding_scale=payload.embedding_scale)
        data = self.resample(payload.sr,data)
        # data = data.detach().numpy()
        file = io.BytesIO()
        write(file,payload.sr,data)
        process_time = time.time() - __t
        yield {'audio': b64encode(file.read()).decode(),'sr':payload.sr,"time":process_time}
        logger.info(f"StyleTTS systhesized release the lock and time take {process_time}")
        self.lock.release()