from .base import TTS
from ReqModule import StyleTTS
from base64 import b64encode
from scipy.io.wavfile import write
import time
import io

class StyleTTS_SYNC(TTS):
    def __init__(self, name: str, rate: int, dtype) -> None:
        super().__init__(name, rate, dtype)
        self.model = StyleTTS()
    def systhesized(self,text,out_sr):
        self.lock.acquire()
        __t = time.time()
        data = self.model.inference(text,diffusion_steps=5,embedding_scale=1)
        data = self.resample(out_sr,data)
        data = data.cpu().numpy()
        file = io.BytesIO()
        write(file,out_sr,data)
        self.lock.release()
        return {'audio': b64encode(file.read()).decode(),'sr':out_sr,"time":time.time() - __t}