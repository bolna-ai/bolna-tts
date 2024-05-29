from .base import TTS
import torch
from TTS.Modules.ReqModule import MeloTTS
from base64 import b64encode
from scipy.io.wavfile import write
import time
import io
from TTS.Logger import configure_logger
from pydantic import BaseModel

logger = configure_logger(__name__)
class MeloTTS_SYNC(TTS):
    def __init__(self) -> None:
        super().__init__("MeloTTS", 24000, torch.float32)
        self.model:MeloTTS = MeloTTS('EN')
        logger.info(f"{self.name} is loaded in {self.model.device}")
        self.speaker_ids = self.model.hps.data.spk2id
    
    class Config(BaseModel):
        text:str
        speacker_id:str
        sr:int
        sdp_ratio:float = 0.2
        noise_scale:float = 0.6
        noise_scale_w:float = 0.8
        speed:float= 1.0

    def systhesized(self,out_sr:int,text:str,speaker_id:str,sdp_ratio:float=0.2,noise_scale:float=0.6,noise_scale_w:float=0.8,speed:float=1.0):
        self.lock.acquire()
        logger.info(f"")
        __t = time.time()
        audio, sr = self.model.synthesize(text,self.speaker_ids[speaker_id],sdp_ratio,sdp_ratio,noise_scale,noise_scale_w,speed)
        self.rate = sr
        audio = torch.from_numpy(audio)
        audio = self.resample(out_sr,audio)
        file = io.BytesIO()
        write(file,out_sr,audio)
        process_time = time.time() - __t
        yield {'audio': b64encode(file.read()).decode(),'sr':out_sr,"time":process_time}
        self.lock.release()
        
