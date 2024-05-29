
from Modules import StyleTTS_SYNC
from Modules import MeloTTS_SYNC
from Modules.base import TTS
from typing import Dict

class TTSOrchestrator:
    def __init__(self) -> None:
        self.__MeloTTS_SYNC = MeloTTS_SYNC()
        self.__StyleTTS_SYNC = StyleTTS_SYNC()

        TTS_PROVIDER:Dict[str:TTS] = {
            "melo":self.__MeloTTS_SYNC,
            "style":self.__StyleTTS_SYNC
        }