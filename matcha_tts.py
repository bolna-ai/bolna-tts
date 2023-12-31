import torch
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.hifigan.denoiser import Denoiser
from pathlib import Path
import soundfile as sf
import numpy as np
from matcha.hifigan.config import v1
from matcha.hifigan.env import AttrDict
from matcha.utils.utils import intersperse
from matcha.text import sequence_to_text, text_to_sequence
import time
import io
import scipy
import asyncio
import queue
import logging
from nltk.tokenize import sent_tokenize


logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchaTTSWrapper:
    def __init__(self, model_name, vocoder_name, device, model_path, vocoder_path, speaking_rate=1.0):
        # Load MatchaTTS model
        print(f"[!] Loading {model_name}!")
        self.model = MatchaTTS.load_from_checkpoint(model_path, map_location=device)
        _ = self.model.eval()
        print(f"[+] {model_name} loaded!")
        
        # Load HiFi-GAN vocoder
        print(f"[!] Loading {vocoder_name}!")
        
        self.vocoder_name = vocoder_name
        if  "hifigan" in vocoder_name:
            h = AttrDict(v1)
            self.vocoder = HiFiGAN(h).to(device)
            self.vocoder.load_state_dict(torch.load(vocoder_path, map_location=device)["generator"])
            _ = self.vocoder.eval()
            self.vocoder.remove_weight_norm()
            self.denoiser = Denoiser(self.vocoder, mode="zeros")

        
        # Set speaking rate
        self.speaking_rate = speaking_rate
        
        # Set device
        self.device = device
        
        self.sample_rate = 22050
        self.request_dict = {}
    
    @torch.inference_mode()
    def process_text(self, text):
        print(f"Input text: {text}")
        x = torch.tensor(
            intersperse(text_to_sequence(text, ["english_cleaners2"]), 0),
            dtype=torch.long,
            device=self.device,
        )[None]
        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=self.device)
        x_phones = sequence_to_text(x.squeeze(0).tolist())
        print(f"Phonetised text: {x_phones[1::2]}")
        return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}

    @torch.inference_mode()
    def _generate(self, text):
        # Process the input text and synthesize the mel-spectrogram
        text_processed = self.process_text(text)

        print(f"[🍵] Whisking Matcha-T(ea)TS for: {text}")

        output = self.model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=10,  
            temperature=0.667, 
            spks=None, 
            length_scale=self.speaking_rate,
        )
        print("Got output")
        # Convert the mel-spectrogram to waveform using the vocoder
        output["waveform"] = self.to_waveform(output['mel'])
        
        # Convert the tensor to a NumPy array
        audio_np = output["waveform"].numpy()

        # Normalize the audio samples to be within the range of int16
        audio_np = np.int16(audio_np * 32767)
        return audio_np
      

    @torch.inference_mode()
    def to_waveform(self, mel):
        #HIFI-GAN-BASED
        print("Hifi gan")
        audio = self.vocoder(mel).clamp(-1, 1)
        audio = self.denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze()
        print("Reduced noise and now, returning")
        return audio.cpu().squeeze()
    
    async def process_chunk(self, chunk_index, chunk, queue):
        # Generate audio for the chunk asynchronously and enqueue the result along with the index
        audio_bytes = self._generate(chunk)
        await queue.put((chunk_index, audio_bytes))


    async def handler(self, request_id, queue, num_chunks):
        audio_dict = {}
        processed_chunks = 0
        while processed_chunks < num_chunks:
            chunk_index, audio_bytes = await queue.get()
            audio_dict[chunk_index] = audio_bytes
            print(f"Chunk {chunk_index} processed.")
            processed_chunks += 1

        final_audio_np = None
        for i in sorted(audio_dict.keys()):
            if final_audio_np is None:
                final_audio_np = audio_dict[i]
            else:
                final_audio_np = np.concatenate((final_audio_np, audio_dict[i]))
            
        buf = io.BytesIO()
        concatenated_audio_buffer = io.BytesIO()
        scipy.io.wavfile.write(concatenated_audio_buffer, rate=22050, data=final_audio_np)
        concatenated_raw_audio_bytes = concatenated_audio_buffer.getvalue()
        self.request_dict[request_id] = concatenated_raw_audio_bytes



    async def generate(self, text, request_id):
        chunks = sent_tokenize(text)
        print(chunks)
        queue = asyncio.Queue()
        process_tasks = [self.process_chunk(index, chunk_text, queue) for index, chunk_text in enumerate(chunks)]
        handler_task = self.handler(request_id, queue, len(chunks))

        overall_start_time = time.time()
        await asyncio.gather(*process_tasks, handler_task)
        return self.request_dict[request_id]
    
    #TODO implement generate stream 