import torch
import torch.nn.functional as F
from torch.autograd import grad


def gradient_penalty(critic, h_s, h_t, cond, p):
    #batch_size = h_s.size(0)
    batch_size = min(h_s.size(0), h_t.size(0))
    device = h_s.device
    alpha = torch.rand(batch_size, 1,1,1).to(device)
    differences = h_t[:batch_size] - h_s[:batch_size]

    # Compute interpolates
    interpolates = h_s[:batch_size] + (alpha) * differences[:batch_size]
    interpolates = interpolates.requires_grad_(True)

    # Get critic output
    preds,_ = critic(interpolates, cond[:batch_size], p)

    # Compute gradients
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]

    # Compute gradient penalty
    gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty
