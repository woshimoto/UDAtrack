import torch
import torch.nn.functional as F


def select_evidence_mask(scores, valid_mask, topk, threshold, fallback_tokens=1):
    """Select threshold-qualified evidence with a hard per-track budget.

    ``threshold`` controls admission and ``topk`` only caps the number of
    admitted tokens. A tiny fallback avoids empty attention inputs when every
    score is below the threshold.
    """
    if scores.shape != valid_mask.shape:
        raise ValueError("scores and valid_mask must have the same shape")

    original_shape = scores.shape
    flat_scores = scores.reshape(-1, scores.shape[-1])
    flat_valid = valid_mask.reshape(-1, valid_mask.shape[-1]).bool()
    flat_keep = torch.zeros_like(flat_valid)

    for row in range(flat_scores.shape[0]):
        valid = flat_valid[row] & torch.isfinite(flat_scores[row])
        candidate_idx = torch.nonzero(
            valid & (flat_scores[row] >= threshold), as_tuple=False
        ).flatten()

        if topk > 0 and candidate_idx.numel() > topk:
            ranked = torch.topk(flat_scores[row, candidate_idx], topk).indices
            candidate_idx = candidate_idx[ranked]

        if candidate_idx.numel() == 0 and fallback_tokens > 0:
            valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
            fallback = min(fallback_tokens, valid_idx.numel())
            if fallback > 0:
                ranked = torch.topk(flat_scores[row, valid_idx], fallback).indices
                candidate_idx = valid_idx[ranked]

        if candidate_idx.numel() > 0:
            flat_keep[row, candidate_idx] = True

    return flat_keep.reshape(original_shape)


def cosine_association_reliability(query, state, valid_mask):
    """Map query-state cosine similarity to [0, 1] for valid tracks."""
    reliability = (F.cosine_similarity(query, state, dim=-1) + 1.0) * 0.5
    return reliability.clamp(0, 1) * valid_mask.to(reliability.dtype)


def box_motion_consistency(current_boxes, previous_boxes, valid_mask, eps=1e-6):
    """Score center displacement relative to the previous box size."""
    current_center = current_boxes[..., :2]
    previous_center = previous_boxes[..., :2]
    previous_size = previous_boxes[..., 2:].clamp_min(eps)
    normalized_delta = (current_center - previous_center) / previous_size
    consistency = torch.exp(-0.5 * normalized_delta.square().sum(dim=-1))
    return consistency * valid_mask.to(consistency.dtype)
