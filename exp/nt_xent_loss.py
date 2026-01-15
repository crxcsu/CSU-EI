import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):

    def __init__(self, device, temperature, alpha_weight):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device

    def softXEnt(self, target, logits):

        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs,
                    norm=True
                    ):
        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
            
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()  # (B, B)
        labels = labels.to(self.device)
        
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / self.temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / self.temperature

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)
        loss = (self.alpha_weight*loss_a + (1-self.alpha_weight)*loss_b)
        return loss

# class NTXentLoss(nn.Module):

#     def __init__(self, device, temperature, alpha_weight):
#         super(NTXentLoss, self).__init__()
#         self.temperature = temperature
#         self.alpha_weight = alpha_weight
#         self.device = device

#     def forward(self, zis, zjs, norm=True):
#         # Get (normalized) hidden1 and hidden2.
#         if norm:
#             zis = F.normalize(zis, p=2, dim=1)
#             zjs = F.normalize(zjs, p=2, dim=1)

#         hidden1, hidden2 = zis, zjs
#         batch_size = hidden1.shape[0]

#         # Labels: diagonal matrix indicating positive pairs
#         labels = torch.arange(start=0, end=batch_size, dtype=torch.long).to(self.device)

#         # Compute logits
#         logits_ab = torch.matmul(hidden1, hidden2.t()) / self.temperature
#         logits_ba = torch.matmul(hidden2, hidden1.t()) / self.temperature

#         # Compute loss using CrossEntropyLoss
#         loss_fn = nn.CrossEntropyLoss()
#         loss_a = loss_fn(logits_ab, labels)
#         loss_b = loss_fn(logits_ba, labels)

#         # Combine losses with alpha_weight
#         loss = self.alpha_weight * loss_a + (1 - self.alpha_weight) * loss_b
#         return loss
