import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dataset import rgb_to_ycbcr
from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.out_nc == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_imgs = batch['gt'], batch['gt']
            imgs = (rgb_to_ycbcr(imgs)[..., 0, :, :]).unsqueeze(-3)
            true_imgs = (rgb_to_ycbcr(true_imgs)[..., 0, :, :]).unsqueeze(-3)

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_imgs = true_imgs.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs, true_imgs)

            if net.out_nc > 1:
                tot += F.cross_entropy(mask_pred, true_imgs).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_imgs).item()
            pbar.update()

    net.train()
    return tot / n_val
