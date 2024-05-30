from torchaudio import transforms
import numpy as np
from threading import Lock
class TTS:
    def __init__(self,name:str,rate:int,dtype) -> None:
        self.name = name
        self.rate = rate
        self.dtype = dtype
        self.lock = Lock() 
    def systhesized(self,config):
        """use locks over here"""
        raise("systhesized function is not implemented")
    def resample(self,rate,wav):
        resample_transform = transforms.Resample(self.rate,rate,dtype=self.dtype)
        return resample_transform(wav).cpu().numpy()