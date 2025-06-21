"""
Provides a function to generate text using LLaDA 1.5.
The model will be downloaded automatically on first use to the current directory.
"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import warnings
import os
import math
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
    """Load the LLaDA 1.5 model and tokenizer if not already loaded."""
    global _model, _tokenizer, _device
    
    if _model is not None and _tokenizer is not None:
        return
    
    _device = _get_device()
    print(f"Loading LLaDA 1.5 model on {_device}...")
    
    cache_dir = "./llada_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    _tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-1.5", 
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    _model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-1.5",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if _device != "cpu" else torch.float32,
        device_map="auto" if _device == "cuda" else None,
        cache_dir=cache_dir
    )
    
    if _device != "cuda":
        _model = _model.to(_device)
    
    _model.eval()
    print("Model loaded successfully!")

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
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
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
def llada_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336, save_history=False):
    """
    Generate text using LLaDA's masked diffusion process.
    
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy - 'low_confidence' (semi-autoregressive) or 'random' (default: 'low_confidence')
        mask_id: The token id of [MASK] is 126336.
        save_history: Whether to save generation history for analysis.
    
    Returns:
        If save_history=False: generated sequence tensor
        If save_history=True: (generated sequence tensor, generation history dict)
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    
    # Initialize history tracking
    history = {
        'states': [],           # List of token states at each step
        'predictions': [],      # List of model's raw predictions at each step
        'finalized_at_step': {},  # Maps token position to step when it was finalized
        'step_info': [],        # Info about each step (block, iteration)
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
                # Save state before this step
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
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if save_history:
                history['predictions'].append(x0.clone())

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
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
            
            # Track which tokens are being finalized
            if save_history:
                newly_finalized = transfer_index[0].cpu().numpy()
                for pos_idx, is_finalized in enumerate(newly_finalized):
                    if is_finalized and pos_idx not in history['finalized_at_step']:
                        history['finalized_at_step'][pos_idx] = global_step
            
            x[transfer_index] = x0[transfer_index]
            global_step += 1

    # Save final state
    if save_history:
        history['states'].append(x.clone())
        history['predictions'].append(x.clone())
        history['step_info'].append({
            'global_step': global_step,
            'block': num_blocks,
            'block_step': 0,
            'block_start': prompt.shape[1] + gen_length,
            'block_end': prompt.shape[1] + gen_length,
            'final': True
        })
        return x, history
    
    return x

def generate_text(prompt, gen_length=128, steps=128, block_length=32, temperature=0.0, cfg_scale=0.0, remasking='low_confidence', save_history=False):
    """
    Generate text using LLaDA 1.5.
    
    Args:
        prompt (str): The input text prompt
        gen_length (int): Maximum length of generated text (default: 128)
        steps (int): Number of diffusion sampling steps (default: 128)
        block_length (int): Block length for semi-autoregressive generation (default: 32)
        temperature (float): Sampling temperature, 0 = deterministic (default: 0.0)
        cfg_scale (float): Classifier-free guidance scale (default: 0.0)
        remasking (str): Remasking strategy - 'low_confidence' (semi-autoregressive) or 'random' (default: 'low_confidence')
        save_history (bool): Whether to save generation history for analysis (default: False)
    
    Returns:
        If save_history=False: Generated text string
        If save_history=True: (Generated text string, generation history dict, tokenizer)
    """
    _load_model()
    
    # Gen_length must be divisible by block_length
    gen_length = ((gen_length + block_length - 1) // block_length) * block_length
    
    # Steps must be divisible by number of blocks
    num_blocks = gen_length // block_length
    steps = ((steps + num_blocks - 1) // num_blocks) * num_blocks
    
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = _tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    input_ids = _tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(_device).unsqueeze(0)
    
    if save_history:
        output, history = llada_generate(
            _model, 
            input_ids, 
            steps=steps, 
            gen_length=gen_length, 
            block_length=block_length, 
            temperature=temperature, 
            cfg_scale=cfg_scale,
            remasking=remasking,
            save_history=True
        )
    else:
        output = llada_generate(
            _model, 
            input_ids, 
            steps=steps, 
            gen_length=gen_length, 
            block_length=block_length, 
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            save_history=False
        )
    
    # Decode only the generated part
    generated_text = _tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    if save_history:
        return generated_text.strip(), history, _tokenizer
    else:
        return generated_text.strip()

def _unload_model():
    """Unload the model and free up memory."""
    global _model, _tokenizer, _device
    
    if _model is not None:
        del _model
        _model = None
    
    if _tokenizer is not None:
        del _tokenizer  
        _tokenizer = None
    
    _device = None
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Model unloaded successfully!")

@torch.no_grad()
def llada_generate_batch(model, prompts, steps=128, gen_length=128, block_length=128, temperature=0.,
                        cfg_scale=0., remasking='low_confidence', mask_id=126336, save_history=False):
    """
    Generate text using LLaDA's masked diffusion process for multiple prompts.
    
    Args:
        model: Mask predictor.
        prompts: A tensor of shape (batch_size, max_prompt_length).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        save_history: Whether to save generation history for analysis.
    
    Returns:
        If save_history=False: generated sequences tensor of shape (batch_size, max_prompt_length + gen_length)
        If save_history=True: (generated sequences tensor, list of generation histories)
    """
    batch_size, max_prompt_length = prompts.shape
    x = torch.full((batch_size, max_prompt_length + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :max_prompt_length] = prompts.clone()

    prompt_index = (x != mask_id)
    
    # Initialize history tracking for batch
    histories = []
    if save_history:
        for i in range(batch_size):
            history = {
                'states': [],           
                'predictions': [],      
                'finalized_at_step': {},  
                'step_info': [],        
                'mask_id': mask_id,
                'prompt_length': (prompts[i] != mask_id).sum().item()
            }
            histories.append(history)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    global_step = 0
    
    for num_block in range(num_blocks):
        block_start = max_prompt_length + num_block * block_length
        block_end = max_prompt_length + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            if save_history:
                # Save state before this step for all samples
                for j in range(batch_size):
                    histories[j]['states'].append(x[j:j+1].clone())
                    histories[j]['step_info'].append({
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
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if save_history:
                for j in range(batch_size):
                    histories[j]['predictions'].append(x0[j:j+1].clone())

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
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
            
            # Track which tokens are being finalized
            if save_history:
                for j in range(batch_size):
                    newly_finalized = transfer_index[j].cpu().numpy()
                    for pos_idx, is_finalized in enumerate(newly_finalized):
                        if is_finalized and pos_idx not in histories[j]['finalized_at_step']:
                            histories[j]['finalized_at_step'][pos_idx] = global_step
            
            x[transfer_index] = x0[transfer_index]
            global_step += 1

    # Save final state
    if save_history:
        for j in range(batch_size):
            histories[j]['states'].append(x[j:j+1].clone())
            histories[j]['predictions'].append(x[j:j+1].clone())
            histories[j]['step_info'].append({
                'global_step': global_step,
                'block': num_blocks,
                'block_step': 0,
                'block_start': max_prompt_length + gen_length,
                'block_end': max_prompt_length + gen_length,
                'final': True
            })
        return x, histories
    
    return x

def generate_texts_batch(prompts, gen_length=128, steps=128, block_length=32, temperature=0.0, 
                        cfg_scale=0.0, remasking='low_confidence', save_history=False, 
                        batch_size=None, unload_after=False):
    """
    Generate text using LLaDA 1.5 for multiple prompts efficiently.
    
    Args:
        prompts (list of str): List of input text prompts
        gen_length (int): Maximum length of generated text (default: 128)
        steps (int): Number of diffusion sampling steps (default: 128)
        block_length (int): Block length for semi-autoregressive generation (default: 32)
        temperature (float): Sampling temperature, 0 = deterministic (default: 0.0)
        cfg_scale (float): Classifier-free guidance scale (default: 0.0)
        remasking (str): Remasking strategy - 'low_confidence' (semi-autoregressive) or 'random' (default: 'low_confidence')
        save_history (bool): Whether to save generation history for analysis (default: False)
        batch_size (int): Maximum batch size for processing. If None, processes all at once (default: None)
        unload_after (bool): Whether to unload the model after processing (default: False)
    
    Returns:
        If save_history=False: List of generated text strings
        If save_history=True: (List of generated text strings, List of generation histories, tokenizer)
    """
    if not prompts:
        return [] if not save_history else ([], [], None)
    
    _load_model()
    
    # Gen_length must be divisible by block_length
    gen_length = ((gen_length + block_length - 1) // block_length) * block_length
    
    # Steps must be divisible by number of blocks
    num_blocks = gen_length // block_length
    steps = ((steps + num_blocks - 1) // num_blocks) * num_blocks
    
    # Process prompts in batches
    all_results = []
    all_histories = [] if save_history else None
    
    # If no batch_size specified, process all at once
    if batch_size is None:
        batch_size = len(prompts)
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Format and tokenize batch
        formatted_prompts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = _tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            formatted_prompts.append(formatted_prompt)
        
        # Tokenize and pad
        tokenized = _tokenizer(formatted_prompts, padding=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].to(_device)
        
        # Generate
        if save_history:
            outputs, batch_histories = llada_generate_batch(
                _model, 
                input_ids, 
                steps=steps, 
                gen_length=gen_length, 
                block_length=block_length, 
                temperature=temperature, 
                cfg_scale=cfg_scale,
                remasking=remasking,
                save_history=True
            )
            all_histories.extend(batch_histories)
        else:
            outputs = llada_generate_batch(
                _model, 
                input_ids, 
                steps=steps, 
                gen_length=gen_length, 
                block_length=block_length, 
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                save_history=False
            )
        
        # Decode generated parts
        for j, (output, input_length) in enumerate(zip(outputs, tokenized['attention_mask'].sum(dim=1))):
            generated_text = _tokenizer.decode(output[input_length:], skip_special_tokens=True)
            all_results.append(generated_text.strip())
    
    if unload_after:
        _unload_model()
    
    if save_history:
        return all_results, all_histories, _tokenizer
    else:
        return all_results
