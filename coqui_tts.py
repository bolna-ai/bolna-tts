import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import logging
import io
import scipy
import numpy as np
from dotenv import load_dotenv
import json
from scipy.io import wavfile

load_dotenv()
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import deepspeed
    use_deepspeed = True
except Exception as e:
    use_deepspeed = False
class CoquiTTSWrapper:
    def __init__(self, sampling_rate = 24000, format = "wav"):

        logger.info("Loading model...")
        if torch.cuda.is_available():
            logging.info("CUDA available, GPU inference used.")
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logging.info("MPS available, but GPU inference won't be used as aten::_fft_r2c' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764.")
            device = torch.device("cpu")
        else:
            logging.info("CUDA and MPS not available, CPU inference used.")
            device = torch.device("cpu")


        config = XttsConfig()
        xtts_base_path = os.getenv("XTTS_BASE_PATH")
        config.load_json(f"{xtts_base_path}config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=xtts_base_path, use_deepspeed=use_deepspeed)
        self.model.to(device)

        with open('./data/voices.json', 'rb') as f:
            self.voices = json.load(f)

    def generate_stream(self, text, language, voice):
        t0 = time.time()
        speaker_embedding = (torch.tensor(self.voices[voice]["speaker_embedding"]).unsqueeze(0).unsqueeze(-1))
        logger.info(f"Got speaker embeddings: {text}")
        gpt_cond_latent = (torch.tensor(self.voices[voice]["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0))

        chunks = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding
        )
        for i, chunk in enumerate(chunks):
            if i ==0:
                chunk_generation_time = time.time() - t0
                chunk_time = time.time()
            else:
                chunk_generation_time = time.time() - chunk_time
                chunk_time = time.time()
            logger.info(f"Received chunk {i} of audio length {chunk.shape[-1]}. Time to generate chunk: {chunk_generation_time}")
            post_processing_start_time = time.time()
            audio_buffer = io.BytesIO()
            audio_np = chunk.squeeze().unsqueeze(0).cpu().numpy()
            # Normalize the audio samples to be within the range of int16
            audio_np = np.int16(audio_np * 32767)
            logger.info(f"Received chunk {i} post processing time {time.time() - post_processing_start_time}")
            yield audio_np.tobytes()


    def generate(self, text, voice, language = "en"):
        logger.info(f"Generating audio for text: {text}")
        speaker_embedding = (torch.tensor(self.voices[voice]["speaker_embedding"]).unsqueeze(0).unsqueeze(-1))
        logger.info(f"Got speaker embeddings: {text}")
        gpt_cond_latent = (torch.tensor(self.voices[voice]["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0))
        out = self.model.inference(text, language, gpt_cond_latent, speaker_embedding)

        audio_numpy = out["wav"]
        audio_numpy = np.interp(audio_numpy, (audio_numpy.min(), audio_numpy.max()), (-1, 1))
        audio_numpy = (audio_numpy * 32767).astype(np.int16)
        audio_buffer = io.BytesIO()
        wavfile.write(audio_buffer, 22050, audio_numpy)  # 22050 is the sample rate, change if different
        raw_audio_bytes = audio_buffer.getvalue()
        return raw_audio_bytes
