import os, sys
import torch, torchaudio
import argparse
import json
from omegaconf import MISSING, OmegaConf,DictConfig
from huggingface_hub import hf_hub_download

os.environ['DISABLE_FLASH_ATTN'] = "1"
from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler
from normalize_lyrics import clean_lyrics

NAME2REPO = {
    "songbloom_full_150s" : "CypressYang/SongBloom",
    "songbloom_full_150s_dpo" : "CypressYang/SongBloom",
    "songbloom_full_240s" : "CypressYang/SongBloom_long",
}

def hf_download(model_name="songbloom_full_150s", local_dir="./cache", **kwargs):
    
    repo_id = NAME2REPO[model_name]
    
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=f"{model_name}.yaml", local_dir=local_dir, **kwargs)
    ckpt_path = hf_hub_download(
        repo_id=repo_id, filename=f"{model_name}.pt", local_dir=local_dir, **kwargs)
    
    
    vae_cfg_path = hf_hub_download(
        repo_id="CypressYang/SongBloom", filename="stable_audio_1920_vae.json", local_dir=local_dir, **kwargs)
    vae_ckpt_path = hf_hub_download(
        repo_id="CypressYang/SongBloom", filename="autoencoder_music_dsp1920.ckpt", local_dir=local_dir, **kwargs)
    
    g2p_path = hf_hub_download(
        repo_id="CypressYang/SongBloom", filename="vocab_g2p.yaml", local_dir=local_dir, **kwargs)
    
    return 
    

    
    


def load_config(cfg_file, parent_dir="./") -> DictConfig:
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda x: os.path.splitext(os.path.basename(x))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: OmegaConf.load(x))
    OmegaConf.register_new_resolver("dynamic_path", lambda x: x.replace("???", parent_dir))
    # cmd_cfg = OmegaConf.from_cli()
    
    file_cfg = OmegaConf.load(open(cfg_file, 'r')) if cfg_file is not None \
                else OmegaConf.create()
    

    return file_cfg



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="songbloom_full_150s")
    parser.add_argument("--local-dir", type=str, default="./cache")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--dtype", type=str, default='float32', choices=['float32', 'bfloat16']) # There appear to be some bugs in FP16
    parser.add_argument("--device", type=str, default='cuda:0' ) # "cpu"
    args = parser.parse_args()

    hf_download(args.model_name, args.local_dir)
    cfg = load_config(f"{args.local_dir}/{args.model_name}.yaml", parent_dir=args.local_dir)
  
    cfg.max_dur = cfg.max_dur + 10
    
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    model = SongBloom_Sampler.build_from_trainer(cfg, strict=False, dtype=dtype, device=device)
    model.set_generation_params(**cfg.inference)
          
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_lines = open(args.input_jsonl, 'r').readlines()
    input_lines = [json.loads(l.strip()) for l in input_lines]
    
    for test_sample in input_lines:
        # print(test_sample)
        idx, lyrics, prompt_wav = test_sample["idx"], test_sample["lyrics"], test_sample["prompt_wav"]
        # lyrics = clean_lyrics(lyrics) # This function can handle some wrong cases of lyrics input (not all)

        prompt_wav, sr = torchaudio.load(prompt_wav)
        if sr != model.sample_rate:
            prompt_wav = torchaudio.functional.resample(prompt_wav, sr, model.sample_rate)
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True).to(dtype)
        prompt_wav = prompt_wav[..., :10*model.sample_rate]
        # breakpoint()
        for i in range(args.n_samples):
            wav = model.generate(lyrics, prompt_wav)
            torchaudio.save(f'{args.output_dir}/{idx}_s{i}.flac', wav[0].cpu().float(), model.sample_rate)


if __name__ == "__main__":
    
    main()
