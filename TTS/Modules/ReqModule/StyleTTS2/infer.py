import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import nltk
nltk.download('punkt')
# load packages
import random
import yaml
import numpy as np
import torchaudio
from nltk.tokenize import word_tokenize
import phonemizer
from StyleTTS2.models import *
from StyleTTS2.utils import *
from StyleTTS2.text_utils import TextCleaner
from StyleTTS2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from StyleTTS2.Utils.PLBERT.util import load_plbert
import os


class StyleTTS:
    def __init__(self):
        self.to_mel:torchaudio.transforms.MelSpectrogram = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean:int = -2 
        self.std:int = 4
        self.RATE = 24000
        self.config = yaml.safe_load(open("Models/LJSpeech/config.yml"))
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')
        self.textclenaer = TextCleaner()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.path = os.path.abspath("")+"/StyleTTS2/"
        
        ASR_config = self.path + self.config.get('ASR_config', False)
        ASR_path = self.path + self.config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)
        # load pretrained F0 model
        F0_path = self.path +self.config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        BERT_path = self.path + self.config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)

        self.model = build_model(recursive_munch(self.config['model_params']), text_aligner, pitch_extractor, plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
        params = params_whole['net']

        for key in self.model:
            if key in params:
                print('%s loaded' % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]
                        new_state_dict[name] = v
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        _ = [self.model[key].eval() for key in self.model]
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )
    def inference(self, text, diffusion_steps=5, embedding_scale=1):
        noise = noise = torch.randn(1,1,256).to(self.device)
        text = text.strip()
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise,
                embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                embedding_scale=embedding_scale).squeeze(0)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu()
