
from TTS.Modules import StyleTTS_SYNC, MeloTTS_SYNC
from TTS.Modules import  DummyTTS_SYNC
from TTS.Modules.base import TTS
from typing import Dict
from pydantic import BaseModel
from TTS.Logger import configure_logger


logger = configure_logger(__name__)

class TTS_PROVIDER:
    def __init__(self,model:TTS,modelclass) -> None:
        self.model:TTS = model
        self.modelclass = modelclass.Config
class TTSOrchestrator:
    def __init__(self,Debug=False) -> None:
        logger.info("starting TTS Orchestrator")
        self.__TTS_PROVIDER:Dict[str:TTS_PROVIDER] = dict()
        self.__RegisterModel(MeloTTS_SYNC)
        if not Debug:
            self.__RegisterModel(DummyTTS_SYNC)
            self.__RegisterModel(StyleTTS_SYNC)
        logger.info(f"models loaded: {list(self.__TTS_PROVIDER.keys())}")
        logger.info("TTS Orchestrator is started")
    def __RegisterModel(self,tts):
        model:TTS = tts()
        temp = TTS_PROVIDER(model,tts)
        self.__TTS_PROVIDER[model.name] = temp
    def getModel(self,model_name)->TTS:
        if self.__TTS_PROVIDER.get(model_name) == None: return None
        return self.__TTS_PROVIDER.get(model_name).model
    def getModelConfig(self,model_name) -> BaseModel:
        if self.__TTS_PROVIDER.get(model_name) == None: return None
        return self.__TTS_PROVIDER.get(model_name).modelclass