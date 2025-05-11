import os

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from transformers import BitsAndBytesConfig

from huggingface_hub import snapshot_download
import torch



def load_quantized_32B(device = 'cuda'): 
    Q_MODEL_NAME = "Qwen/Qwen3-32B"
    qbnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                       # or load_in_8bit=True
        bnb_4bit_quant_type="nf4",               # only for 4-bit
        bnb_4bit_use_double_quant=True,          # only for 4-bit
        bnb_4bit_compute_dtype=torch.float16,    # internal compute dtype
    )
    Q_tokenizer = AutoTokenizer.from_pretrained(
        Q_MODEL_NAME,
    )
    Q_model = AutoModelForCausalLM.from_pretrained(
        Q_MODEL_NAME,
        quantization_config=qbnb_config,
    ).to(device).eval()
    return Q_model, Q_tokenizer

def load_quantized_14B(device = 'cuda'): 
    Q_MODEL_NAME = "Qwen/Qwen3-14B"
    qbnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                       # or load_in_8bit=True
        bnb_4bit_quant_type="nf4",               # only for 4-bit
        bnb_4bit_use_double_quant=True,          # only for 4-bit
        bnb_4bit_compute_dtype=torch.float16,    # internal compute dtype
    )
    Q_tokenizer = AutoTokenizer.from_pretrained(
        Q_MODEL_NAME
    )
    Q_model = AutoModelForCausalLM.from_pretrained(
        Q_MODEL_NAME,
        quantization_config=qbnb_config,
    ).to(device).eval()
    return Q_model, Q_tokenizer

def load_small_model(device = 'cuda'):
    Q_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    Q_tokenizer = AutoTokenizer.from_pretrained(
        Q_MODEL_NAME
    )
    Q_model = AutoModelForCausalLM.from_pretrained(
        Q_MODEL_NAME
    ).to(device).eval()
    return Q_model, Q_tokenizer


def load_llada_base(device = "cuda"):
    model_name = "GSAI-ML/LLaDA-8B-Base"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

def load_llada_instruct(device = "cuda"):
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer