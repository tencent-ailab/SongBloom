import torch
from torch import nn
import typing as tp
import torchaudio
import einops
from abc import ABC, abstractmethod


class AbstractVAE(ABC, nn.Module):

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        ...

    @property
    @abstractmethod
    def orig_sample_rate(self) -> int:
        ...
    

    @property
    @abstractmethod
    def channel_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def split_bands(self) -> int:
        ...

    @property
    @abstractmethod
    def input_channel(self) -> int:
        ...


    def encode(self, wav) -> torch.Tensor:
        ...
        
    def decode(self, latents) -> torch.Tensor:
        ...


from .autoencoders import create_autoencoder_from_config, AudioAutoencoder
class StableVAE(AbstractVAE):
    def __init__(self, vae_ckpt, vae_cfg, sr=48000) -> None:
        super().__init__()
        import json
        with open(vae_cfg) as f:
            config = json.load(f)
        self.vae: AudioAutoencoder = create_autoencoder_from_config(config)

        # Use map_location='cpu' on Apple MPS to avoid device mismatch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            state_dict = torch.load(vae_ckpt, map_location='cpu')['state_dict']
        else:
            state_dict = torch.load(vae_ckpt)['state_dict']
        self.vae.load_state_dict(state_dict)

        self.sample_rate = sr
        self.rsp48k = torchaudio.transforms.Resample(sr, self.orig_sample_rate) if sr != self.orig_sample_rate else nn.Identity()
       
    @torch.no_grad()
    def encode(self, wav: torch.Tensor, sample=True) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        wav = self.rsp48k(wav)
        if wav.shape[-1] < 2048:
            return torch.zeros((wav.shape[0], self.channel_dim, 0), device=wav.device, dtype=wav.dtype)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        if wav.shape[1] == 1:
            wav = wav.repeat(1, self.vae.in_channels, 1)
        latent = self.vae.encode_audio(wav) # B, 64, T
        return latent
            

        
    def decode(self, latents: torch.Tensor, **kwargs):
        # B, 64, T
        with torch.no_grad():
            audio_recon = self.vae.decode_audio(latents, **kwargs)
            
        return audio_recon
        
    @property
    def frame_rate(self) -> float:
        return float(self.vae.sample_rate) / self.vae.downsampling_ratio

    @property
    def orig_sample_rate(self) -> int:
        return self.vae.sample_rate

    @property
    def channel_dim(self) -> int:
        return self.vae.latent_dim

    @property
    def split_bands(self) -> int:
        return 1
    
    @property
    def input_channel(self) -> int:
        return self.vae.in_channels
