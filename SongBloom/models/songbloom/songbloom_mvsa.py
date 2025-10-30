from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import logging
import math
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
import tqdm

from ..base.utils import create_norm_fn
from ..base.sample import sample_top_k, sample_top_p, multinomial
from ..musicgen.modules.streaming import StreamingModule
from ..musicgen.conditioners import (
    get_condition_fuser,
    get_conditioner_provider,
     ConditionType,
     ConditioningProvider,
     ConditionFuser,
     AttributeDropout,
     ClassifierFreeGuidanceDropout,
     ConditioningAttributes,
     WavCondition,
    JointEmbedCondition
)

from ..musicgen.get_backend import get_backend

from ..transformer import ContinuousTransformer as DiT_block
from ..musicldm.musicldm_dit import FourierFeatures
from ..musicldm.inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler, sample_discrete_euler_with_temperature

ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]

@dataclass
class DiTAROutput:
    ar_logit: torch.Tensor  
    ar_target: torch.Tensor
    nar_pred: torch.Tensor
    nar_target: torch.Tensor
    nar_t: torch.Tensor



class MVSA_DiTAR(StreamingModule):
    """
    Multiple skeleton embedding, single compressed vae latent 
    eg. V1 V2 V3 A1-3 V4 V5 V6 A4-6
        V -> cross entropy (skeleton)
        A -> local-DiT uncompress -> (A1-3 -> E1 E2 E3)

    Args:
        StreamingModule (_type_): _description_
    """

    def __init__(self, condition_provider_cfg, fuser_cfg, 
                 block_size: int = 32, dim: int = 1024, num_heads: int = 8,
                 num_pitch: int = 128, hidden_scale: int = 4, lm_layers: int = 16,
                 norm: str = 'layer_norm', pre_norm: bool = False,
                 backend='llama',init_std: float=0.02, 
                 # ======================
                 latent_dim: int = 64, diff_layers: int = 8,
                 time_cond_type: tp.Literal['adaLM', "prepend"] = "prepend", 
                 timestep_features_dim: int = 256,
                 diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
                 timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform",
                 rotary_base_val=10000, h_dropout: float = None, 
                 # ======================
                 cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}
                 ):
        super().__init__()

        self.condition_provider = get_conditioner_provider(condition_provider_cfg)
        self.fuser = get_condition_fuser(fuser_cfg)
        
        self.dim = dim  
        self.latent_dim = latent_dim
        self.block_size = block_size
            
        self.cfg_coef = cfg_coef
        self.h_dropout = h_dropout if h_dropout is not None else 0.
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)


        # Build AR lm
        self.num_pitch = num_pitch + 1 # self.num_pitch = <EOS>, self.num_pitch+1 = special
        self.skeleton_emb = nn.Embedding(self.num_pitch + 1, dim)
        self.bos_token = nn.Parameter(torch.empty(dim).normal_(mean=0.0, std=init_std), requires_grad=True)

        
        # self.lm_type = lm_type
        self.backend = backend
        
        if self.backend == 'llama':
            self.ar_transformer = get_backend('llama', 
                                        dim, num_heads, lm_layers, hidden_scale,init_std=init_std, rope_theta=rotary_base_val)
            self.ar_transformer.gradient_checkpointing_enable()
        elif self.backend == 'bart':
            self.cross_encoder = get_backend('bart_enc',
                                        dim, num_heads, lm_layers // 4, hidden_scale,init_std=init_std)
            self.ar_transformer = get_backend('bart_dec',
                                        dim, num_heads, lm_layers, hidden_scale,init_std=init_std)
        else:
            raise NotImplementedError(f"Illegal backend: {self.backend}!")
        

        self.skeleton_classifier =  nn.Sequential(nn.Linear(dim, dim, bias=False), 
                                                    nn.SiLU(),
                                                    nn.Linear(dim, self.num_pitch),)
        
        self.pre_norm: tp.Optional[nn.Module] = None
        if pre_norm:
            self.pre_norm = create_norm_fn(norm, dim)
        self.reset_streaming()
        
        # Build NAR DiT
        self.block_conv = nn.Sequential(
            Rearrange("b d (n s) -> b n (s d)", s=self.block_size),
            nn.Linear(self.block_size * latent_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.project_in = nn.Linear(latent_dim, dim) if latent_dim != dim else nn.Identity()
        self.project_out = nn.Linear(dim, latent_dim) if latent_dim != dim else nn.Identity()
        
        self.timestep_features_dim = timestep_features_dim
        self.time_cond_type = time_cond_type 
        assert self.time_cond_type in ['adaLN', "prepend"]
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        self.time_cond_type = time_cond_type
        self.nar_dit = DiT_block(
            dim=dim,
            depth=diff_layers,
            dim_heads= dim // num_heads,
            rotary_pos_emb=True,
            cross_attend=False,
            causal=False,
            ff_kwargs={"dim_ff": dim * hidden_scale, "no_bias": True},
            global_cond_dim=self.dim if self.time_cond_type=="adaLN" else None,
            rotary_base_val = 20000,
            # init_std=init_std
        )
        self.nar_dit.gradient_checkpointing_enable()
        
        self.diffusion_objective = diffusion_objective
        self.timestep_sampler = timestep_sampler
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        
        self.init_weights(init_std=init_std)
            
        


    @property
    def special_token_id(self) -> int:
        return self.num_pitch


    @property
    def eos_token_id(self) -> int:
        return self.num_pitch-1

    def forward(self, x_sketch, x_latent, x_len, condition_tensors) -> DiTAROutput:
        '''
        only for train: lm_forward + diffusion_forward (random_t)
        x_sketch: (B,T) # T % block_sz == 0 (no <eos> token) padded with <eos>
        x_latent: (B, D_{in}, T)
        '''
        # AR 
        assert torch.all(x_len % self.block_size == 0), f"{x_len}"
        block_num = x_len // self.block_size
        
        sketch_emb = self.skeleton_emb(x_sketch)
        latent_emb = self.block_conv(x_latent)

        B, T, D = sketch_emb.shape
        
        lm_input = rearrange(torch.cat([rearrange(sketch_emb, "b (n s) d -> b n s d", s=self.block_size), 
                            latent_emb.unsqueeze(dim=2)], dim=2), "b n s d -> b (n s) d")
        lm_input = torch.cat([self.bos_token.reshape(1,1,-1).expand(B,-1,-1),
                            lm_input], dim=1) #add <sos>
        
        new_seq_len = x_len + block_num + 1
        
        ar_target = F.pad(x_sketch, (0,1), value=self.eos_token_id)
        for b,l in enumerate(x_len):
            ar_target[b, l+1:] = self.special_token_id # 用来mask掉多余的eos
        

    
        lm_out = self.lm_forward(lm_input, condition_tensors)
        
        
        indices = torch.arange(lm_out.shape[1])
        h_ind = indices[(indices+1) % (self.block_size+1) == 0]
        not_h_ind = indices[(indices+1) % (self.block_size+1) != 0]
        
        x_sketch_logit = self.skeleton_classifier(lm_out[:, not_h_ind])
        
        # NAR (h + prev_block)
        h_pad = lm_out[:, h_ind] # B, N, D        
        h = torch.cat([hh[:hl] for hh, hl in zip(h_pad, block_num)], dim=0) 
        block_semantic = rearrange(sketch_emb, "b (n s) d -> b n s d", s=self.block_size) # B, N, 32, D 
        current_block_semantic = torch.cat([bb[:bl] for bb, bl in zip(block_semantic, block_num)], dim=0) 
        
        if self.training: # for CFG
            drop_h_idx = torch.rand((h.shape[0], 1), device=h.device) < self.h_dropout
            h = torch.masked_fill(h, drop_h_idx, 0)
            # current_block_semantic = torch.masked_fill(current_block_semantic, drop_h_idx.unsqueeze(-1), 0)
            
            drop_s_idx = torch.rand((current_block_semantic.shape[0], 1), device=current_block_semantic.device) < self.h_dropout
            current_block_semantic = torch.masked_fill(current_block_semantic, drop_s_idx.unsqueeze(-1), 0)
        
        with torch.no_grad():
            block_latent = rearrange(x_latent, "b d (n s) -> b n s d", s=self.block_size) # B, N, 32, D
            current_block = torch.cat([bb[:bl] for bb, bl in zip(block_latent, block_num)], dim=0) 
            prev_block = torch.cat([bb[:bl] for bb, bl in zip(F.pad(block_latent, (0,0,0,0,1,0)), block_num)], dim=0)
       
            # b_indices = torch.randperm(block_latent.shape[0])[:B*16]
            # h, current_block, prev_block = h[b_indices], current_block[b_indices], prev_block[b_indices]
            
            orig_type = x_latent.dtype
            with torch.cuda.amp.autocast(enabled=False):
                if self.timestep_sampler == "uniform":
                    # Draw uniformly distributed continuous timesteps
                    t = self.rng.draw(h.shape[0])[:, 0].to(device=h.device, dtype=h.dtype)
                elif self.timestep_sampler == "logit_normal":
                    t = torch.sigmoid(torch.randn(h.shape[0], device=h.device, dtype=h.dtype))
                elif self.timestep_sampler == "trunc_logit_normal":
                    # Draw from logistic truncated normal distribution
                    from ..musicldm.musicldm_pl import truncated_logistic_normal_rescaled
                    t = truncated_logistic_normal_rescaled(h.shape[0]).to(h.device)
                    # Flip the distribution
                    t = 1 - t
                    
                # Calculate the noise schedule parameters for those timesteps
                if self.diffusion_objective == "v":
                    alphas, sigmas = get_alphas_sigmas(t)
                elif self.diffusion_objective == "rectified_flow":
                    alphas, sigmas = 1-t, t
                # Combine the ground truth data and the noise
                alphas = alphas[:, None, None]
                sigmas = sigmas[:, None, None]
                noise = torch.randn_like(current_block)
                noised_inputs = current_block * alphas + noise * sigmas
                if self.diffusion_objective == "v": # (a_t - a_{t-1})x_0 + (b_t-b_{t-1}) e = -b x_0 + a e
                    targets = noise * alphas - current_block * sigmas
                elif self.diffusion_objective == "rectified_flow": #||(XT-X0) - p(x_t, t)||      
                    targets = noise - current_block
    
        nar_output = self.diffusion_forward(noised_inputs.to(orig_type), t.to(orig_type), h, current_block_semantic, prev_block)

        return DiTAROutput(
            ar_logit=x_sketch_logit,
            ar_target=ar_target, 
            nar_pred=nar_output,
            nar_target=targets.to(orig_type),
            nar_t=t
        )



    def lm_forward(self, sequence, condition_tensors: tp.Optional[ConditionTensors] = None) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        B, T, D = sequence.shape
        if self.pre_norm:
            sequence = self.pre_norm(sequence.to(self.pre_norm.weight.data.dtype))
            
        input_, cross_attention_input = self.fuser(sequence, condition_tensors)

        transformer_input = {
            "inputs_embeds":input_,
            "use_cache": self._is_streaming, 
            "past_key_values": self._streaming_state.get('past_key_values', None),
        }
        if self.backend == 'bart': # TODO infer 的时候这个玩意不用重复算
            # TODO attention_mask
            cross_attention_input = self.cross_encoder(inputs_embeds=cross_attention_input) 
            transformer_input["encoder_hidden_states"] = cross_attention_input.last_hidden_state

        output = self.ar_transformer(**transformer_input)
        if self._is_streaming:
            self._streaming_state['past_key_values'] = output.past_key_values
        out = output.last_hidden_state
             

            
        if len(self.fuser.fuse2cond['prepend']) > 0:
            out = out[:, -T:, :]

        return out




    def diffusion_forward(self, 
                x: torch.Tensor,
                t: torch.Tensor, # B,
                h: torch.Tensor,
                s: torch.Tensor, # B, self.block_size, D
                history_x: torch.Tensor,
                cfg_coef: float = None) -> torch.Tensor:

        if cfg_coef is not None:
            # only for infer
            assert not self.training # only for inference
            x = torch.cat([x,x], dim=0)
            t = torch.cat([t,t], dim=0)
            h = torch.cat([h,torch.zeros_like(h)], dim=0)
            s = torch.cat([s,torch.zeros_like(s)], dim=0)
            history_x = torch.cat([history_x,history_x], dim=0)
            
        B, T, _ = x.shape
        
        input_ = self.project_in(torch.cat([history_x, x], dim=1))
        # print(h.shape, s.shape, input_.shape)
        input_ = torch.cat([h.unsqueeze(1), s, input_], dim=1)
        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]))# (b, embed_dim)
        # breakpoint()
        if self.time_cond_type == "prepend":
            input_ = torch.cat([timestep_embed.unsqueeze(1), input_], dim=1)

        transformer_input = {
            "x": input_,
            "global_cond": timestep_embed if self.time_cond_type == "adaLN" else None}
        
        output = self.nar_dit(**transformer_input)

        # remove the prefix from the model outputs
        output = output[:, -T:, :]
        output = self.project_out(output)

        if cfg_coef is not None:
            cond_output, uncond_output = torch.chunk(output, 2, dim=0)
            output = uncond_output + (cond_output - uncond_output) * cfg_coef
        
        return output  # [B, T, D]



    def _sample_next_block(self,
                           sequence: torch.Tensor,
                           prev_latents: torch.Tensor, 
                           condition_tensors: tp.Optional[ConditionTensors] = None,
                           cfg_coef: tp.Optional[tp.Union[float, tp.List[float]]] = None,
                           steps: int = 50,
                           dit_cfg_type: str = 'h',
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           diff_temp: float = 1.0, 
                           top_k: int = 0,
                           top_p: float = 0.0,
                           penalty_token_pool: tp.Optional[list] = None) -> torch.Tensor:
        # infer: lm next_token -> (if % block_sz == 0) infer diff
        # 1. sample sketch (lm) -> 2. sample latent (lm+diff)
        sequence = sequence.clone()
        
        if isinstance(cfg_coef, tp.Iterable):
            assert len(cfg_coef) == 2
            cfg_coef_lm, cfg_coef_diff = cfg_coef
        else:
            cfg_coef_lm, cfg_coef_diff = cfg_coef, cfg_coef
        
        B = sequence.shape[0]
        # import pdb; pdb.set_trace()

        if condition_tensors:
            # Preparing for CFG, predicting both conditional and unconditional logits.
            sequence = torch.cat([sequence, sequence], dim=0)
        
        
        # ############### decode sketch #########################
        next_tokens = []
        next_token_embs = []
            
        for k in range(self.block_size):
            if self._is_streaming and k > 0:
                lm_inp = sequence[:,-1:]
            else:
                lm_inp = sequence
                
            lm_out = self.lm_forward(
                lm_inp,
                condition_tensors=condition_tensors)
            next_pitch_logit = self.skeleton_classifier(lm_out[:, -1:]) # B, 1, card

            if condition_tensors:
                cond_logit, uncond_logit = next_pitch_logit.split(B, dim=0)  
                next_pitch_logit = uncond_logit + (cond_logit - uncond_logit) * cfg_coef_lm

            # add penalty to pre-sampled tokens
            if penalty_token_pool is not None and len(penalty_token_pool) > 0: # B, T
                for b in range(B):
                    # q_count = torch.bincount(penalty_token_pool)
                    q_count = torch.bincount(torch.unique(penalty_token_pool[b]))
                    tmp = min(q_count.shape[-1], self.num_pitch - 1) 
                    next_pitch_logit[b, -1, :tmp] /= (1.1 ** q_count[:tmp])
                    
            # sample k
            if use_sampling and temp > 0.0:
                probs = torch.softmax(next_pitch_logit  / temp, dim=-1)
                if top_p > 0.0:
                    next_token = sample_top_p(probs, p=top_p)
                elif top_k > 0:
                    next_token = sample_top_k(probs, k=top_k)
                else:
                    next_token = multinomial(probs, num_samples=1)
                next_token = next_token.squeeze(-1)
            else:
                next_token = torch.argmax(next_pitch_logit, dim=-1) # B, 1
            if penalty_token_pool is not None and len(penalty_token_pool) > 0: # B, T
                penalty_token_pool = torch.cat([penalty_token_pool, next_token], dim=-1)[:,1:]
            next_token_emb = self.skeleton_emb(next_token) #B, 1, d
            next_tokens.append(next_token)
            next_token_embs.append(next_token_emb)
            
            if condition_tensors:
                doubled_next_emb = torch.cat([next_token_emb, next_token_emb], dim=0)
                sequence = torch.cat([sequence, doubled_next_emb], dim=1)
            else:
                sequence = torch.cat([sequence, next_token_emb], dim=1)
            
        next_tokens = torch.cat(next_tokens, dim=1)
        next_token_embs = torch.cat(next_token_embs, dim=1)
        
        # ############### decode latent ###########################
        # 这里求h虽然double了 但是没用classifier-free guidance
        if self._is_streaming:
            lm_inp = sequence[:,-1:]
        else:
            lm_inp = sequence
                
        lm_out = self.lm_forward(
            lm_inp,
            condition_tensors=condition_tensors)
        
        h = lm_out[:,-1] 

        noise = torch.randn((B, self.block_size, self.latent_dim), device=h.device, dtype=h.dtype)
        
        assert dit_cfg_type in ['h', 'global', 'none']
        """
        global: same cfg setting as next-token-prediction
        none: no cfg
        h: no cfg during ar-stage and apply cfg via ar output
        """
        if condition_tensors:
            if dit_cfg_type == 'global':
                noise = torch.cat([noise, noise], dim=0)
                prev_latents = torch.cat([prev_latents, prev_latents], dim=0)
                semantic_embs = torch.cat([next_token_embs, next_token_embs], dim=0)
            else:
                h, _ = h.chunk(2, dim=0)      
                semantic_embs = next_token_embs

        
        if self.diffusion_objective == "v":
            next_latent = sample(self.diffusion_forward, noise, steps=steps, eta=0, h=h, s=semantic_embs, history_x=prev_latents, 
                                 cfg_coef=(cfg_coef_diff if dit_cfg_type=='h' else None))
        elif self.diffusion_objective == "rectified_flow":
            # next_latent = sample_discrete_euler(self.diffusion_forward, noise, steps=steps, h=h, s=semantic_embs, history_x=prev_latents, 
            #                                     cfg_coef=(cfg_coef_diff if dit_cfg_type=='h' else None))
            next_latent = sample_discrete_euler_with_temperature(self.diffusion_forward, noise, steps=steps, temperature=diff_temp, h=h, s=semantic_embs, history_x=prev_latents, 
                                                cfg_coef=(cfg_coef_diff if dit_cfg_type=='h' else None))
        if condition_tensors and dit_cfg_type == 'global':
            cond_next_latent, uncond_next_latent = torch.chunk(next_latent, 2, dim=0)
            next_latent = uncond_next_latent + (cond_next_latent - uncond_next_latent) * cfg_coef_diff
            
        latent_emb = self.block_conv(next_latent.transpose(1,2))

        next_block_seq = torch.cat([next_token_embs, latent_emb], dim=1) # B, self.block_size+1, d
        
        return next_tokens, next_latent, next_block_seq
        
            
        
    @torch.no_grad()
    def generate(self,
                 prompt: tp.Optional[torch.Tensor] = None,
                 conditions: tp.List[ConditioningAttributes] = [],
                 cfg_coef: tp.Optional[tp.Union[float, tp.List[float]]] = None,
                 steps=50,
                 dit_cfg_type: str = 'h',
                 max_frames: int = 1500, # 60 * 25
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 diff_temp: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 0.0,
                 penalty_repeat: bool = False,
                 penalty_window: int = 50) -> torch.Tensor: 
        assert not self.training, "generation shouldn't be used in training mode."

        B = len(conditions)
        assert B==1, "currently  do not support batch decoding"
        null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
        conditions = conditions + null_conditions
        tokenized = self.condition_provider.tokenize(conditions)
        condition_tensors = self.condition_provider(tokenized)          
        
        
        sequence = self.bos_token.reshape(1,1,-1).expand(B, 1, -1)
        if prompt is not None:
            # TODO 
            raise NotImplementedError
            # sequence = torch.cat([sequence, prompt])

            
        prev_blocks = torch.zeros((B, self.block_size, self.latent_dim), device=sequence.device, dtype=sequence.dtype)
        latent_seq, token_seq = None, None
            
        with self.streaming():
            prog_bar = tqdm.tqdm()
            while True:
                if token_seq is None or not penalty_repeat:
                    penalty_token_pool = None
                else:
                    penalty_token_pool = token_seq[: ,-penalty_window:]
                    if penalty_token_pool.shape[-1] < penalty_window:
                        penalty_token_pool = F.pad(penalty_token_pool, (penalty_window - penalty_token_pool.shape[-1], 0), value=self.eos_token_id)
                next_tokens, next_latent, next_block_seq = self._sample_next_block(sequence[:, -1: ], prev_blocks, condition_tensors, 
                                                                                   cfg_coef=cfg_coef, steps=steps, dit_cfg_type=dit_cfg_type,
                                                                                    use_sampling=use_sampling, temp=temp, diff_temp=diff_temp,
                                                                                    top_k=top_k, top_p=top_p,
                                                                                    penalty_token_pool=penalty_token_pool)
                
                if (next_tokens == self.eos_token_id).any() or sequence.shape[1] > max_frames  / self.block_size * (self.block_size+1):
                    break
                
                latent_seq = next_latent if latent_seq is None else torch.cat([latent_seq, next_latent], dim=1) # B,T, D
                token_seq = next_tokens if token_seq is None else torch.cat([token_seq, next_tokens], dim=1) # B,T
                sequence = torch.cat([sequence, next_block_seq], dim=1)
                prev_blocks = next_latent
                
                prog_bar.update(self.block_size)
                
                
        if latent_seq is None:
            latent_seq = prev_blocks
        return latent_seq.transpose(1,2), token_seq    
        
        
    
    
    def init_weights(self, init_std=0.02):
        
        def _init_weights(module, init_std=0.02):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=init_std)
                # torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=init_std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
                    
        self.apply(partial(_init_weights, init_std=init_std))