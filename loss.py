import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ContrastiveLoss(nn.Module):
    def __init__(self, device, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, vi, ui): # vi: image patches, ui: time-series patches
        batch_size, num_patches, dim = vi.shape
        vi = vi / vi.norm(dim=-1, keepdim=True)
        ui = ui / ui.norm(dim=-1, keepdim=True)
        logits = torch.matmul(ui, vi.transpose(-1, -2)) / self.temperature 
        labels = torch.arange(num_patches).expand(batch_size, num_patches).to(self.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.transpose(-1, -2), labels)
        loss = (loss_i + loss_t) / 2 
        return loss
