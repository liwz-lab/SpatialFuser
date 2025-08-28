import torch
import torch.nn.functional as F


def reconstruction_loss(original, X_decode):
    if original.is_sparse:
        return F.mse_loss(original.to_dense(), X_decode)
        # return torch.abs(original.to_dense() - X_decode).mean()
        # return Cauchy_loss(1, original.to_dense(), X_decode)
    else:
        return F.mse_loss(original, X_decode)
        # return torch.abs(original - X_decode).mean()
        # return Cauchy_loss(1, original, X_decode)

def overall_direction_loss(embedding1, embedding2):
    ST_normed = F.normalize(embedding1, dim=-1)
    SP_normed = F.normalize(embedding2, dim=-1)
    cos_sim = torch.matmul(ST_normed, SP_normed.T)
    return 1-cos_sim.mean()


def improved_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    distance_function = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")

    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    torch._check(
        a_dim == p_dim and p_dim == n_dim,
        lambda: (
            f"The anchor, positive, and negative tensors are expected to have "
            f"the same number of dimensions, but got: anchor {a_dim}D, "
            f"positive {p_dim}D, and negative {n_dim}D inputs"
        ),
    )

    if distance_function is None:
        distance_function = torch.pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    # The distance swap is described in the paper "Learning shallow
    # convolutional feature descriptors with triplet losses" by V. Balntas, E.
    # Riba et al.  If True, and if the positive example is closer to the
    # negative example than the anchor is, swaps the positive example and the
    # anchor in the loss computation.
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)
    loss = torch.clamp_min(margin + dist_pos - dist_neg, 0)
    return dist_pos.mean() + _apply_loss_reduction(loss, reduction)


def _apply_loss_reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:  # reduction == "none"
        return loss