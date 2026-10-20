import torch

CLASS_NAMES = {1: "NCR", 2: "ED", 3: "ET", 4: "RC"}


@torch.no_grad()
def compute_dice_stats(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class intersection and total (|pred| + |target|) voxel counts for one batch.

    Accumulate these over the whole dataset and pass the sums to dice_from_stats,
    so that batches without a given class do not inflate the score.
    """
    pred_classes = torch.argmax(pred, dim=1)

    intersections = []
    totals = []
    for cls_idx in CLASS_NAMES:
        pred_mask = pred_classes == cls_idx
        target_mask = target == cls_idx

        intersections.append((pred_mask & target_mask).sum())
        totals.append(pred_mask.sum() + target_mask.sum())

    return torch.stack(intersections).double(), torch.stack(totals).double()


def dice_from_stats(intersection: torch.Tensor, total: torch.Tensor) -> dict[str, float]:
    """Global Dice per class from accumulated counts; a class absent everywhere gets 1.0."""
    scores = {}
    for i, cls_name in enumerate(CLASS_NAMES.values()):
        if total[i] == 0:
            score = 1.0
        else:
            score = (2.0 * intersection[i] / total[i]).item()
        scores[cls_name] = round(score, 4)

    scores["mean"] = round(sum(scores.values()) / len(CLASS_NAMES), 4)
    return scores
