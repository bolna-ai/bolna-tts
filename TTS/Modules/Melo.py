from .base import TTS
import torch
from ReqModule import MeloTTS
from base64 import b64encode
from scipy.io.wavfile import write
import time
import io

class MeloTTS_SYNC(TTS):
    def __init__(self) -> None:
        super().__init__("MeloTTS", 24000, torch.float32)
        self.model:MeloTTS = MeloTTS('EN')
    def systhesized(self,out_sr:int,text:str,speaker_id:str,sdp_ratio:float=0.2,noise_scale:float=0.6,noise_scale_w:float=0.8,speed:float=1.0):
        self.lock.acquire()
        __t = time.time()
        audio, sr = self.model.synthesize(text,speaker_id,sdp_ratio,sdp_ratio,noise_scale,noise_scale_w,speed)
        self.rate = sr
        audio = torch.from_numpy(audio)
        audio = self.resample(out_sr,audio)
        file = io.BytesIO()
        write(file,out_sr,audio)
        self.lock.release()
        return {'audio': b64encode(file.read()).decode(),'sr':out_sr,"time":time.time() - __t}
