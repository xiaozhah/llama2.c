import torch
from dataclasses import dataclass
from model import Attention, precompute_freqs_cis

@dataclass
class ModelArgs:
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: int = 8
    max_seq_len: int = 2048
    dropout: float = 0.1

def test_attention():
    args = ModelArgs()
    attention = Attention(args)
    
    # dummy input
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, args.dim)
    
    freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
    freqs_cos = freqs_cos[:seq_len]
    freqs_sin = freqs_sin[:seq_len]

    # Forward pass (fake data)
    print(x.shape, freqs_cos.shape, freqs_sin.shape)
    output = attention(x, freqs_cos, freqs_sin)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, args.dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    
    print("Attention module test passed!")

if __name__ == "__main__":
    test_attention()