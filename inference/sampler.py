import torch
import torch.nn.functional as F
from typing import List
from bpe import BPETokenizer
from model import TransformerModel


@torch.no_grad()
def sample(
    model: TransformerModel,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    device: str = "cpu"
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        if input_tensor.size(1) >= model.embedder.positional_embedding.positional_embedding.size(1):
            break  # max length

        logits = model(input_tensor)  # (1, T, vocab_size)
        next_logits = logits[:, -1, :] / temperature  # (1, vocab_size)

        # Top-k sampling
        if top_k > 0:
            values, indices = torch.topk(next_logits, k=top_k)
            next_logits = torch.full_like(next_logits, float('-inf')).scatter(1, indices, values)

        # Top-p (nucleus) sampling
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            cutoff = cumulative_probs > top_p
            if cutoff.any():
                cutoff_index = cutoff[0].nonzero(as_tuple=False)[0]
                sorted_logits[0, cutoff_index:] = float('-inf')
                next_logits = torch.full_like(next_logits, float('-inf')).scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        input_tensor = torch.cat([input_tensor, next_token], dim=1)

    generated_ids = input_tensor[0].tolist()
    return tokenizer.decode(generated_ids)

