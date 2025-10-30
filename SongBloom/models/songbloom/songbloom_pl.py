
from functools import partial
import typing as tp
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
import numpy as np
import random
from omegaconf import OmegaConf
import copy
import lightning as pl

import os, sys

from ..musicgen.conditioners import WavCondition, JointEmbedCondition, ConditioningAttributes
from ..vae_frontend import StableVAE
from .songbloom_mvsa import MVSA_DiTAR
from ...g2p.lyric_common import key2processor, symbols, LABELS


os.environ['TOKENIZERS_PARALLELISM'] = "false"


class SongBloom_PL(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # 关闭自动优化
        # self.automatic_optimization = False

        self.cfg = cfg

        # Build VAE
        self.vae = StableVAE(**cfg.vae).eval()
        assert self.cfg.model['latent_dim'] == self.vae.channel_dim

            
        self.save_hyperparameters(cfg)
        if self.vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False
                
        # Build DiT
        model_cfg = OmegaConf.to_container(copy.deepcopy(cfg.model), resolve=True)
        for cond_name in model_cfg["condition_provider_cfg"]:
            if model_cfg["condition_provider_cfg"][cond_name]['type'] == 'audio_tokenizer_wrapper':
                model_cfg["condition_provider_cfg"][cond_name]["audio_tokenizer"] = self.vae
                model_cfg["condition_provider_cfg"][cond_name]["cache"] = False
        
        
        self.model = MVSA_DiTAR(**model_cfg)
        # print(self.model)
        





####################################

class SongBloom_Sampler:    
    
    def __init__(self, compression_model: StableVAE, diffusion: MVSA_DiTAR, lyric_processor_key,
                 max_duration: float, prompt_duration: tp.Optional[float] = None):
        self.compression_model = compression_model
        self.diffusion = diffusion
        self.lyric_processor_key = lyric_processor_key
        self.lyric_processor = key2processor.get(lyric_processor_key) if lyric_processor_key is not None else lambda x: x
        # import pdb; pdb.set_trace()

        assert max_duration is not None
        self.max_duration: float = max_duration
        self.prompt_duration = prompt_duration
        
        
        self.device = next(iter(diffusion.parameters())).device
        self.generation_params: dict = {}
        # self.set_generation_params(duration=15)  # 15 seconds by default
        self.set_generation_params(cfg_coef=1.5, steps=50, dit_cfg_type='h',
                                   use_sampling=True, top_k=200, max_frames=self.max_duration * 25)
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None

    @classmethod
    def build_from_trainer(cls, cfg, strict=True, dtype=torch.float32, device=None):
        model_light = SongBloom_PL(cfg)
        incompatible = model_light.load_state_dict(torch.load(cfg.pretrained_path, map_location='cpu'), strict=strict)
        
        lyric_processor_key = cfg.train_dataset.lyric_processor
    
        print(incompatible)
        
        model_light = model_light.eval()  
        if device is None:
            model_light = model_light.cuda()
        else:
            model_light = model_light.to(device)
        
        
        model = cls(
            compression_model = model_light.vae,
            diffusion = model_light.model.to(dtype=dtype),
            lyric_processor_key = lyric_processor_key,
            max_duration = cfg.max_dur,
            prompt_duration = cfg.sr * cfg.train_dataset.prompt_len
            
        )
        model.set_generation_params(**cfg.inference)
        return model
        
    @property
    def frame_rate(self) -> float:
        """Roughly the number of AR steps per seconds."""
        return self.compression_model.frame_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated audio."""
        return self.compression_model.sample_rate


    def set_generation_params(self, **kwargs):
        """Set the generation parameters."""
        self.generation_params.update(kwargs)

    # Mulan Inference
    @torch.no_grad()
    def generate(self, lyrics, prompt_wav) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """ Generate samples conditioned on text and melody.
        """
        # breakpoint()
        assert prompt_wav.ndim == 2
        if self.prompt_duration is not None:
            prompt_wav = prompt_wav[..., :self.prompt_duration]
            
        attributes, _ = self._prepare_tokens_and_attributes(conditions={"lyrics": [self._process_lyric(lyrics)], "prompt_wav": [prompt_wav]}, 
                                                                        prompt=None, prompt_tokens=None)

        # breakpoint()
        print(self.generation_params)
        latent_seq, token_seq = self.diffusion.generate(None, attributes, **self.generation_params)
        # print(token_seq)
        # audio_recon = self.compression_model.decode(latent_seq.float())
        audio_recon = self.compression_model.decode(latent_seq.float(), chunked=True)
        
        return audio_recon
    

    def _process_lyric(self, input_lyric):
        if self.lyric_processor_key == 'pinyin':
            processed_lyric = self.lyric_processor(input_lyric)
        else:
            processed_lyric = []
            check_lyric = input_lyric.split(" ")
            for ii in range(len(check_lyric)):
                if check_lyric[ii] not in symbols and check_lyric[ii] not in LABELS.keys() and len(check_lyric[ii]) > 0:
                    new = self.lyric_processor(check_lyric[ii])
                    check_lyric[ii] = new
            processed_lyric = " ".join(check_lyric)
        
        return processed_lyric
    
    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            conditions: tp.Dict[str, tp.List[tp.Union[str, torch.Tensor]]],
            prompt: tp.Optional[torch.Tensor],
            prompt_tokens: tp.Optional[torch.Tensor] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        batch_size = len(list(conditions.values())[0])
        assert batch_size == 1
        # breakpoint()
        attributes = [ConditioningAttributes() for _ in range(batch_size)]
        for k in self.diffusion.condition_provider.conditioners:
            conds = conditions.pop(k, [None for _ in attributes])
            for attr, cond in zip(attributes, conds):
                if self.diffusion.condition_provider.conditioner_type[k] == 'wav':
                    if cond is None:
                        attr.wav[k] = WavCondition(
                            torch.zeros((1, 1, 1), device=self.device),
                            torch.tensor([0], device=self.device).long(),
                            sample_rate=[self.sample_rate],
                            path=[None])  
                    else:
                        attr.wav[k] = WavCondition(
                            cond.to(device=self.device).unsqueeze(0), # 1,C,T .mean(dim=0, keepdim=True)
                            torch.tensor([cond.shape[-1]], device=self.device).long(),
                            sample_rate=[self.sample_rate],
                            path=[None])  
                elif self.diffusion.condition_provider.conditioner_type[k] == 'text':
                    attr.text[k] = cond
                elif self.diffusion.condition_provider.conditioner_type[k] == 'joint_embed':
                    if cond is None or isinstance(cond, str):
                        attr.joint_embed[k] = JointEmbedCondition(
                            torch.zeros((1, 1, 1), device=self.device),
                            [cond],
                            torch.tensor([0], device=self.device).long(),
                            sample_rate=[self.sample_rate],
                            path=[None])  
                    elif isinstance(cond, torch.Tensor):
                        attr.joint_embed[k] = JointEmbedCondition(
                            cond.to(device=self.device).mean(dim=0, keepdim=True).unsqueeze(0),
                            [None], 
                            torch.tensor([cond.shape[-1]], device=self.device).long(),
                            sample_rate=[self.sample_rate],
                            path=[None])  
                    else:
                        raise NotImplementedError
        assert conditions == {}, f"Find illegal conditions: {conditions}, support keys: {self.lm.condition_provider.conditioners}"
        # breakpoint()
        print(attributes)
        
        if prompt_tokens is not None:
            prompt_tokens = prompt_tokens.to(self.device)
            assert prompt is None
        elif prompt is not None:
            assert len(attributes) == len(prompt), "Prompt and nb. attributes doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens = self.compression_model.encode(prompt)
        else:
            prompt_tokens = None

        return attributes, prompt_tokens

