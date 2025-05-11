import os
import torch
import numpy as np
import torch.nn.functional as F

from rich import print

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from transformers import BitsAndBytesConfig

from huggingface_hub import snapshot_download

from utils import *

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, 
             tokenizer=None, scorer_model = None, scorer_tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence', 'random' or 'custom'. If custom, uses the scorer model to remask. 
        mask_id: The toke id of [MASK] is 126336.
    '''


    @torch.no_grad()
    def scorer_confidence(seq, scorer_model):
        """
        Returns P(seq[i] | seq[:i]) for every i ≥ 1.

        seq : (B, L)  – the *filled* candidate sequence.
        out : (B, L)  – probability of each token under scorer model.
                        out[:, 0] is set to 1.0 because the model never
                        predicts the first token (it’s conditioned on BOS).
        """
        logits = scorer_model(seq).logits          # (B, L, V)

        # The logit at t predicts token at t, so compare logits[:, t-1] with seq[:, t]
        probs  = torch.softmax(logits, dim=-1).to(torch.float64)  # (B, L, V)

        # Shift: targets are seq[:, 1:], predictors are probs[:, :-1]
        tgt      = seq[:, 1:]                      # (B, L-1)
        predProb = torch.gather(
                    probs[:, :-1],              # (B, L-1, V)
                    -1,
                    tgt.unsqueeze(-1)           # (B, L-1, 1)
                ).squeeze(-1)                   # (B, L-1)

        # Pad the first position (no prediction available) with probability 1
        bos_pad  = torch.ones(seq.size(0), 1, dtype=predProb.dtype,
                            device=predProb.device)
        return torch.cat([bos_pad, predProb], dim=1)   # (B, L)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
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

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'custom':
                if scorer_model is None:
                    raise ValueError("pass scorer_model=scoring‑LM (e.g. Qwen) for custom remasking")
                
                ## make a copy for the scorer model 
                vocab_size = scorer_tokenizer.vocab_size
                filled_x = torch.where(mask_index, x0, x)
                x0_p = scorer_confidence(filled_x, scorer_model)
                
            else:
                raise NotImplementedError(remasking)
            
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            
            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

# new confidence transfer generation
# instead of only adding on k tokens each step
# you expand your "candidates" by k tokens 
# so you can possibly mask out previously kept tokens 
@ torch.no_grad()
def generate_new_conf(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, 
             tokenizer=None, scorer_model = None, scorer_tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''

    @torch.no_grad()
    def scorer_confidence(seq, scorer_model):
        """
        Returns P(seq[i] | seq[:i]) for every i ≥ 1.

        seq : (B, L)  – the *filled* candidate sequence.
        out : (B, L)  – probability of each token under Qwen.
                        out[:, 0] is set to 1.0 because the model never
                        predicts the first token (it’s conditioned on BOS).
        """
        logits = scorer_model(seq).logits          # (B, L, V)

        # The logit at t predicts token at t, so compare logits[:, t-1] with seq[:, t]
        probs  = torch.softmax(logits, dim=-1).to(torch.float64)  # (B, L, V)

        # Shift: targets are seq[:, 1:], predictors are probs[:, :-1]
        tgt      = seq[:, 1:]                      # (B, L-1)
        predProb = torch.gather(
                    probs[:, :-1],              # (B, L-1, V)
                    -1,
                    tgt.unsqueeze(-1)           # (B, L-1, 1)
                ).squeeze(-1)                   # (B, L-1)

        # Pad the first position (no prediction available) with probability 1
        bos_pad  = torch.ones(seq.size(0), 1, dtype=predProb.dtype,
                            device=predProb.device)
        return torch.cat([bos_pad, predProb], dim=1)   # (B, L)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
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

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'custom':
                if scorer_model is None:
                    raise ValueError("pass scorer_model=scoring‑LM (e.g. Qwen) for custom remasking")
                
                vocab_size = scorer_tokenizer.vocab_size
                filled_x = torch.where(mask_index, x0, x)
                replace_mask = filled_x == 220
                filled_x[replace_mask] = torch.randint(0, vocab_size, filled_x[replace_mask].shape, device=filled_x.device)
                x0_p = scorer_confidence(filled_x, scorer_model)
                
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            
            confidence = torch.where(mask_index, x0_p, -np.inf)
            confidence = x0_p.clone()
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                zero_count = (mask_index[j] == 0).sum()
                to_transfer = num_transfer_tokens[j, i] + zero_count - prompt_index[j].sum()
                if to_transfer > confidence[j].shape[0]:
                    print("Warning: to_transfer exceeds available tokens, adjusting to maximum available.")
                    to_transfer = min(to_transfer, confidence[j].shape[0])
                
                non_prompt_conf = confidence[j].clone()
                non_prompt_conf[prompt_index[j]] = -float("inf")
                _, select_index = torch.topk(non_prompt_conf, k=to_transfer)


                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            x[~transfer_index & (~prompt_index)] = mask_id
            x0[~transfer_index & (~prompt_index)] = mask_id

    return x



### Original generation method
@ torch.no_grad()
def OG_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, 
             tokenizer=None, scorer_model = None, scorer_tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
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

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'custom':
                raise NotImplementedError("remasking='custom' not supported in OG_generate")
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

# Interpolate between low_confidence and scorer model
@ torch.no_grad()
def generate_interp(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, 
             tokenizer=None, scorer_model = None, scorer_tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random' or 'custom'
        mask_id: The toke id of [MASK] is 126336.
    '''
    @torch.no_grad()
    def scorer_confidence(seq, scorer_model):
        """
        Returns P(seq[i] | seq[:i]) for every i ≥ 1.

        seq : (B, L)  – the *filled* candidate sequence.
        out : (B, L)  – probability of each token under Qwen.
                        out[:, 0] is set to 1.0 because the model never
                        predicts the first token (it’s conditioned on BOS).
        """
        logits = scorer_model(seq).logits          # (B, L, V)

        # The logit at t predicts token at t, so compare logits[:, t-1] with seq[:, t]
        probs  = torch.softmax(logits, dim=-1).to(torch.float64)  # (B, L, V)

        # Shift: targets are seq[:, 1:], predictors are probs[:, :-1]
        tgt      = seq[:, 1:]                      # (B, L-1)
        predProb = torch.gather(
                    probs[:, :-1],              # (B, L-1, V)
                    -1,
                    tgt.unsqueeze(-1)           # (B, L-1, 1)
                ).squeeze(-1)                   # (B, L-1)

        # Pad the first position (no prediction available) with probability 1
        bos_pad  = torch.ones(seq.size(0), 1, dtype=predProb.dtype,
                            device=predProb.device)
        return torch.cat([bos_pad, predProb], dim=1)   # (B, L)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
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

            # Algorithm : for the first half of the steps always generate using low_confidence 
            # for the next half, use thhe custom remasking
            current_progress = i / steps 
            rand_float = torch.rand(1).item()

            if rand_float < 1 - current_progress or remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'custom':
                if scorer_model is None:
                    raise ValueError("pass scorer_model=scoring‑LM (e.g. Qwen) for custom remasking")
                
                vocab_size = scorer_tokenizer.vocab_size
                filled_x = torch.where(mask_index, x0, x)
                x0_p = scorer_confidence(filled_x, scorer_model)
                
            else:
                raise NotImplementedError(remasking)
            
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

# Piecewise generation
# For the first proportion of the steps, use low confidence remasking 
# For the remainder of the steps use custom remasking 
@ torch.no_grad()
def generate_piecewise(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, 
             tokenizer=None, scorer_model = None, scorer_tokenizer=None, proportion_naive = 0.5):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random' or 'custom'
        mask_id: The toke id of [MASK] is 126336.
    '''
    @torch.no_grad()
    def scorer_confidence(seq, scorer_model):
        """
        Returns P(seq[i] | seq[:i]) for every i ≥ 1.

        seq : (B, L)  – the *filled* candidate sequence.
        out : (B, L)  – probability of each token under Qwen.
                        out[:, 0] is set to 1.0 because the model never
                        predicts the first token (it’s conditioned on BOS).
        """
        logits = scorer_model(seq).logits          # (B, L, V)

        # The logit at t predicts token at t, so compare logits[:, t-1] with seq[:, t]
        probs  = torch.softmax(logits, dim=-1).to(torch.float64)  # (B, L, V)

        # Shift: targets are seq[:, 1:], predictors are probs[:, :-1]
        tgt      = seq[:, 1:]                      # (B, L-1)
        predProb = torch.gather(
                    probs[:, :-1],              # (B, L-1, V)
                    -1,
                    tgt.unsqueeze(-1)           # (B, L-1, 1)
                ).squeeze(-1)                   # (B, L-1)

        # Pad the first position (no prediction available) with probability 1
        bos_pad  = torch.ones(seq.size(0), 1, dtype=predProb.dtype,
                            device=predProb.device)
        return torch.cat([bos_pad, predProb], dim=1)   # (B, L)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
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

            # Algorithm : for the first proportion of the steps always generate using low_confidence 
            # for the remaining, use the custom remasking
            if i < proportion_naive * steps or remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'custom':
                if scorer_model is None:
                    raise ValueError("pass scorer_model=scoring‑LM (e.g. Qwen) for custom remasking")
                
                filled_x = torch.where(mask_index, x0, x)
                x0_p = scorer_confidence(filled_x, scorer_model)
                
            else:
                raise NotImplementedError(remasking)
            
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

@torch.no_grad()
def generate_speculative(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, 
             tokenizer=None, scorer_model = None, scorer_tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''

    @torch.no_grad()
    def decode_step_into_words(step, show=True):
        decoded = tokenizer.decode(step[0].tolist(), skip_special_tokens=True)
        if show:
            print(repr(decoded))
        return decoded

    def show_masked(before_mask, after_mask, canidate_masks, braces=("{", "}")):
        ids = before_mask[0].tolist()

        mask_set      = set((after_mask       == mask_id).nonzero(as_tuple=True)[1].tolist())
        candidate_set = set((canidate_masks   == mask_id).nonzero(as_tuple=True)[1].tolist())

        # Decode one token at a time so boundaries stay visible
        tokens = [
            tokenizer.decode([tid],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
            for tid in ids
        ]

        wrapped = []
        for i, tok in enumerate(tokens):
            tok_disp = tok.replace("\n", r"\n")  # render literal "\n"

            if i in mask_set:
                if i in candidate_set:  # masked *and* candidate → red + underline
                    wrapped.append(f"[red underline]{tok_disp}[/red underline]")
                else:                   # masked only → red
                    wrapped.append(f"[red]{tok_disp}[/red]")
            else:
                if i in candidate_set:  # candidate only → green + underline
                    wrapped.append(f"[green underline]{tok_disp}[/green underline]")
                else:                   # plain → green
                    wrapped.append(f"[green]{tok_disp}[/green]")

        print("".join(wrapped))
        return wrapped

    @torch.no_grad()
    def scorer_confidence(seq, scorer_model):
        """
        Returns P(seq[i] | seq[:i]) for every i ≥ 1.

        seq : (B, L)  – the *filled* candidate sequence.
        out : (B, L)  – probability of each token under Qwen.
                        out[:, 0] is set to 1.0 because the model never
                        predicts the first token (it’s conditioned on BOS).
        """

        # this is our sliding window thing!
        logits = scorer_model(seq).logits          # (B, L, V)
        # print(logits.shape)
        # # embed()
        # # # os._exit(1)

        # The logit at t predicts token at t, so compare logits[:, t-1] with seq[:, t]
        probs  = torch.softmax(logits, dim=-1).to(torch.float64)  # (B, L, V)

        # Shift: targets are seq[:, 1:], predictors are probs[:, :-1]
        tgt      = seq[:, 1:]                      # (B, L-1)
        predProb = torch.gather(
                    probs[:, :-1],              # (B, L-1, V)
                    -1,
                    tgt.unsqueeze(-1)           # (B, L-1, 1)
                ).squeeze(-1)                   # (B, L-1)

        # Pad the first position (no prediction available) with probability 1
        bos_pad  = torch.ones(seq.size(0), 1, dtype=predProb.dtype,
                            device=predProb.device)
        return torch.cat([bos_pad, predProb], dim=1)   # (B, L)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    def do_step(x, remasking="custom", threshold=5e-7):

        x = x.clone() # TODO TODO get RID of this for speed??
        prev_x = x.clone()
       
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

        def low_confidence_remasking():
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            return x0_p

        if i < 0.5 * steps or remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        elif remasking == 'custom':
            if scorer_model is None:
                raise ValueError("pass scorer_model=scoring‑LM (e.g. Qwen) for custom remasking")
            
            filled_x = torch.where(mask_index, x0, x)
            x0_p = scorer_confidence(filled_x, scorer_model)
            
        else:
            raise NotImplementedError(remasking)
        
        # def get_transfers_from_p(x0_p):
        x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        # we want to find the index of the first non-inf confidence value below a given threshold


        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            transfer_index[j, select_index] = True

        # transfer_index = get_transfers_from_p(x0_p)
        x[transfer_index] = x0[transfer_index]

        ### END GENERATE

        ##########################3
        # decode_step_into_words(x0)
        show_masked(before_mask=x0, after_mask=x, canidate_masks=prev_x)
        # os._exit(1)

        return x

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        threshold = 1e-1
        for i in range(steps):
            print(f"\nSTEP {i} ------------------------------")
            prev_x = x.clone()
            x = do_step(x, remasking="custom", threshold=threshold)

            if(torch.equal(x, prev_x)):
                threshold /= 1.02


    return x
        

def main():
    device = 'cuda:0'
    CACHE_DIR  = "/scratch/gpfs/jh1161/model-caches/LLaDA-8B-Base"

    Q_model, Q_tokenizer = load_small_model(device)

    model, tokenizer = load_llada_instruct(device)

    prompt = "What is a Tate module over an elliptic curve?"
    prompts = [prompt] * 1

    for prompt in prompts:

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        out = generate(model, input_ids, steps=64, gen_length=64, block_length=64, temperature=0.5, cfg_scale=0., 
        remasking='low_confidence',
        tokenizer=tokenizer, scorer_model=Q_model, scorer_tokenizer=Q_tokenizer)

        print("\n\nRESPONSE:", tokenizer.batch_decode(out[:, :], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
