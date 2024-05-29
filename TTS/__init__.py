
from TTS.Modules import StyleTTS_SYNC
from TTS.Modules import MeloTTS_SYNC
from TTS.Modules.base import TTS
from typing import Dict
from TTS.Logger import configure_logger


logger = configure_logger(__name__)
class TTSOrchestrator:
    def __init__(self) -> None:
        logger.info("starting TTS Orchestrator")
        # self.__MeloTTS_SYNC = MeloTTS_SYNC()
        self.__StyleTTS_SYNC = StyleTTS_SYNC()
        self.TTS_PROVIDER:Dict[str:TTS] = {
            # "melo":self.__MeloTTS_SYNC,
            "style":self.__StyleTTS_SYNC
        }
        logger.info(f"models loaded: {list(self.TTS_PROVIDER.keys())}")
        logger.info("TTS Orchestrator is started")
    