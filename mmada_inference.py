"""
Provides a function to generate text using MMaDA (Multimodal Large Diffusion Language Model).
The model will be loaded from local directory to avoid configuration issues.
Based on the MMaDA implementation from Gen-Verse/MMaDA-8B-MixCoT.
"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import warnings
import os
import sys

# Add local modules directory to path for LLaDA configuration files
sys.path.insert(0, './mmada_modules')

warnings.filterwarnings("ignore")

_model = None
_tokenizer = None
_device = None

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def _load_model():
    """Load the MMaDA model and tokenizer if not already loaded."""
    global _model, _tokenizer, _device
    
    if _model is not None and _tokenizer is not None:
        return
    
    _device = _get_device()
    print(f"Loading MMaDA model on {_device}...")
    
    # Use local model directory with LLaDA configuration files
    model_path = "./local_mmada_model"
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Local MMaDA model directory not found at {model_path}. Please run setup first.")
    
    _tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    
    _model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if _device != "cpu" else torch.float32,
        device_map="auto" if _device == "cuda" else None
    )
    
    if _device != "cuda":
        _model = _model.to(_device)
    
    _model.eval()
    print("MMaDA model loaded successfully!")

def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because MMaDA employs a linear noise schedule (similar to LLaDA),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@torch.no_grad()
def mmada_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336, save_history=False):
    """
    Generate text using MMaDA's masked diffusion process.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    
    # Initialize history tracking
    history = {
        'states': [],           
        'predictions': [],      
        'finalized_at_step': {},  
        'step_info': [],        
        'mask_id': mask_id,
        'prompt_length': prompt.shape[1]
    } if save_history else None

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    global_step = 0
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            if save_history:
                history['states'].append(x.clone())
                history['step_info'].append({
                    'global_step': global_step,
                    'block': num_block,
                    'block_step': i,
                    'block_start': block_start,
                    'block_end': block_end
                })
            
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if save_history:
                history['predictions'].append(x0.clone())

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            if save_history:
                newly_finalized = transfer_index[0].cpu().numpy()
                for pos_idx, is_finalized in enumerate(newly_finalized):
                    if is_finalized and pos_idx not in history['finalized_at_step']:
                        history['finalized_at_step'][pos_idx] = global_step
            
            x[transfer_index] = x0[transfer_index]
            global_step += 1

    if save_history:
        history['states'].append(x.clone())

    return (x, history) if save_history else x

def generate_text(prompt, gen_length=128, steps=128, block_length=32, temperature=0.0, 
                  cfg_scale=0.0, remasking='low_confidence', save_history=False):
    """Generate text using MMaDA model."""
    _load_model()
    
    inputs = _tokenizer(prompt, return_tensors="pt", padding=False, truncation=False).to(_device)
    input_ids = inputs["input_ids"]
    
    if save_history:
        generated_ids, history = mmada_generate(
            _model, input_ids, steps=steps, gen_length=gen_length, 
            block_length=block_length, temperature=temperature,
            cfg_scale=cfg_scale, remasking=remasking, save_history=True
        )
        generated_text = _tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text, history, _tokenizer
    else:
        generated_ids = mmada_generate(
            _model, input_ids, steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=temperature,
            cfg_scale=cfg_scale, remasking=remasking, save_history=False
        )
        generated_text = _tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

def _unload_model():
    """Unload the model to free memory."""
    global _model, _tokenizer, _device
    
    if _model is not None:
        del _model
        _model = None
        
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
        
    _device = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("MMaDA model unloaded successfully!")

def generate_texts_batch(prompts, gen_length=128, steps=128, block_length=32, temperature=0.0, 
                        cfg_scale=0.0, remasking='low_confidence', save_history=False, 
                        batch_size=None, unload_after=False):
    """Generate text for multiple prompts in batches."""
    _load_model()
    
    if batch_size is None:
        batch_size = len(prompts)
    
    results = []
    
    try:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch_prompts:
                result = generate_text(
                    prompt, gen_length=gen_length, steps=steps,
                    block_length=block_length, temperature=temperature,
                    cfg_scale=cfg_scale, remasking=remasking, save_history=save_history
                )
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    finally:
        if unload_after:
            _unload_model()
    
    return results
