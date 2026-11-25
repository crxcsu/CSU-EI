import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn

class NTXentLoss(nn.Module):
    def __init__(self, device, temperature, alpha_weight, use_distributed=1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device
        self.use_distributed = use_distributed

    def softXEnt(self, target, logits):
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def gather_features(self, features):
        if self.use_distributed <= 1:
            return features
        return torch.cat(torch.distributed.nn.all_gather(features), dim=0)

    def forward(self, zis, zjs, norm=True):
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        hidden1, hidden2 = zis, zjs
        hidden1_all = self.gather_features(hidden1)
        hidden2_all = self.gather_features(hidden2)

        batch_size = hidden1.shape[0]              
        global_batch_size = hidden1_all.shape[0] 
        rank = dist.get_rank() if self.use_distributed > 1 else 0

        labels_idx = torch.arange(start=0, end=batch_size, dtype=torch.int64).to(self.device) + rank * batch_size
        labels = F.one_hot(labels_idx, num_classes=global_batch_size).float()

        logits_ab = torch.matmul(hidden1, hidden2_all.T) / self.temperature
        logits_ba = torch.matmul(hidden2, hidden1_all.T) / self.temperature

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)
        return self.alpha_weight * loss_a + (1 - self.alpha_weight) * loss_b