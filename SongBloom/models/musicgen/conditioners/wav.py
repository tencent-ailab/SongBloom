
from .base import *
import omegaconf
from ...vae_frontend import AbstractVAE

def pad_to_fix_length(x, max_len, pad_value=0.):
    bsz, seq_len = x.shape[:2]
    if seq_len >= max_len:
        return x[:, :max_len]
    else:
        pad_len = max_len - seq_len
        pad_tensor = torch.full((bsz, pad_len, *x.shape[2:]), pad_value, dtype=x.dtype, device=x.device)
        padded_tensor = torch.cat([x, pad_tensor], dim=1)
        return padded_tensor

class AudioTokenizerConditioner(WaveformConditioner):
    def __init__(self, output_dim, audio_tokenizer, cache=False, max_len=None):
        super().__init__(output_dim, output_dim)
        self.max_len = max_len
        self.use_cache = cache
        
        self.__dict__['tokenizer'] = audio_tokenizer
        # breakpoint()
        
        # TODO if cached and not load vae, receive a dict instead
        if isinstance(self.tokenizer, dict):
            self.tokenizer = omegaconf.DictConfig(self.tokenizer)
            self.code_depth = self.tokenizer.channel_dim
            
            
        elif isinstance(self.tokenizer, AbstractVAE):
            self.tokenizer_tp = "vae"
            if self.use_cache:
                self.code_depth = self.tokenizer.channel_dim
            else:
                self.code_depth = 1 # TODO 强制把输入channel设成1了 self.tokenizer.input_channel
            self.output_proj = nn.Identity() if self.output_dim == self.tokenizer.channel_dim \
                            else nn.Linear(self.tokenizer.channel_dim, self.output_dim, bias=False)
                            
        else:
            raise NotImplementedError
        
        
    def forward(self, x: WavCondition):
        wav, lengths, *_ = x
        B = wav.shape[0]
        wav = wav.reshape(B, self.code_depth, -1)
        # print(wav.shape)
        # import torchaudio
        # torchaudio.save("/apdcephfs_cq7/share_1297902/common/erichtchen/shixisheng/cyy/project/music_generation_repo/core/models/musicgen/conditioners/111.wav", wav[0].cpu(), 48000)
        if self.tokenizer_tp == "vae":
            if self.use_cache:
                audio_latents = wav.transpose(-1,-2)
            else:
                self.tokenizer.to(device=self.output_proj.weight.device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        audio_latents = self.tokenizer.encode(wav.float()).transpose(-1,-2)
                    # print('transform wav to vae')
            audio_latents = self.output_proj(audio_latents.to(self.output_proj.weight.dtype))

        # print(audio_latents.shape)
        if self.max_len is not None:
            audio_latents = pad_to_fix_length(audio_latents, self.max_len, 0.)
                    
        if lengths is not None:
            lengths = torch.round(lengths.float() * audio_latents.shape[1] / wav.shape[-1])
            mask = length_to_mask(lengths, max_len=audio_latents.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones((B, audio_latents.shape[1]), device=audio_latents.device,dtype=torch.int)

        audio_latents = audio_latents * mask[..., None]
        
        return audio_latents, mask
     

