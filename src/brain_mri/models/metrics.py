import torch


@torch.no_grad()
def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> dict[str, float]:
    pred_classes = torch.argmax(pred, axis=1)
    
    scores = {}
    class_names = {1: "NCR", 2: "ED", 3: "ET"}
    total_dice = 0.0

    # 
    for cls_idx, cls_name in class_names.items():
        pred_mask = (pred_classes == cls_idx).float()
        target_mask = (target == cls_idx).float()

        intersection = (pred_mask * target_mask).sum().item()
        total = pred_mask.sum().item() + target_mask.sum().item()

        if total == 0:
            score = 1
        else:
            score = (2 * intersection) / (total + 1e-6)

        scores[cls_name] = round(score, 4)
        total_dice += score

    scores['mean'] = round(total_dice / len(class_names), 4)
    return scores

