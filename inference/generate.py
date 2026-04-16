import torch
import torch.nn.functional as F


def generate(model, tokenizer, start_text, max_length=20):
    """Autoregressively sample ``max_length`` new tokens after ``start_text``."""
    model.eval()
    device = next(model.parameters()).device

    token_ids = tokenizer.encode(start_text)
    if not token_ids:
        raise ValueError("start_text must produce at least one token (build_vocab first).")

    current = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(current)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current = torch.cat([current, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(current[0].tolist())
