from threading import Thread
from fastapi import WebSocket
from uuid import uuid4
from TTS.Logger import  configure_logger
from TTS import TTS
from asyncio.queues import Queue
from TTS import TTSOrchestrator
from pydantic import BaseModel
import asyncio
import json



logger = configure_logger(__name__)
class WebsocketClient(Thread):
    def __init__(self, TTS_Orchestrator:TTSOrchestrator, websocket:WebSocket) -> None:
        super().__init__(target=None)

        self.name = str(uuid4())

        self.websocket:WebSocket = websocket
        self.__internal_queue:Queue = Queue()
        
        self.TTS_Orchestrator:TTSOrchestrator = TTS_Orchestrator
        logger.info(f"A thread name: {self.name} is created")
        
        self.model:str = None
        self.samplerate:int = None
        self.__model:TTS = None

        self.loop = asyncio.new_event_loop()
    
    async def __recever(self):
        logger.info("start the recever")
        response = (await self.websocket.receive())
        logger.info(response)
        response = json.loads(response)
        """
        init response 
        model: [MeloTTS,StyleTTS]
        sample_rate: 8000 ~ 16000 any thing
        """
        # self.model = response['model']
        # self.samplerate = response['sample_rate']
        # self.__model = TTSOrchestrator.getModel(self.model)
        # logger.info(f"Thread {self.name} get init config {self.mode} and {self.samplerate}")
        while True:
            try:
                response = await self.websocket.receive()
                """
                    for MELO TTS
                        text: hello world
                        model: MeloTTS
                        sample_rate: 8000
                        speacker_id: EN-US
                        sdp_ratio: 0.2
                        noise_scale: 0.6
                        noise_scale_w: 0.8
                        speed: 1.0

                    for Style TTS
                        text: hello world
                        model: MeloTTS
                        sample_rate: 8000
                        diffusion_steps: 5
                        embedding_scale: 1
                    
                    for Dummy TTS
                        text: hello world
                        model: DummyTTS
                        sample_rate: 8000
                        speacker_id: EN-US
                        sdp_ratio: 0.2
                        noise_scale: 0.6
                        noise_scale_w: 0.8
                        speed: 1.0

                """
                logger.info(response)
                # config:BaseModel = TTSOrchestrator.getModelConfig(self.model)
                # response["sr"] = self.samplerate
                # config.parse_obj(response)
                # self.__internal_queue.put(config)
            except Exception as e:
                logger.info(e)
                logger.info(f"thread is stoping {self.name}")
                break
    async def __process(self):
        logger.info("start the process")
        while True:
            try:
                config = await self.__internal_queue.get()
                output = self.__model.systhesized(config=config)
                await self.websocket.send_json(output)
            except Exception as e:
                logger.info(e)
                logger.info(f"thread is stoping {self.name}")
                break
    
    def run(self):
        logger.info("THREAD STARTED")
        task = self.loop.create_task(self.__process())
        self.loop.run_until_complete(self.__recever())
        self.loop.run_until_complete(task)
        logger.info("THREAD ENDED")
    


            

