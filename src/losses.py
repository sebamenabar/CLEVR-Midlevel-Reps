import torch
import torch.nn.functional as F

def bce_fill(logits, val):
    return F.binary_cross_entropy_with_logits(
        logits, torch.empty_like(logits).fill_(val), reduction="none"
    )