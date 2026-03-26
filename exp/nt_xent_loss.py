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

        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss
    
    def gather_features(self, features):
        """
        Gather features from all GPUs across the distributed process group.
        (Using standard no_grad DDP gather for local_loss contrastive learning).
        """
        if not (dist.is_available() and dist.is_initialized()):
            return features
            
        world_size = dist.get_world_size()
        
        with torch.no_grad():
            gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
            dist.all_gather(gathered_features, features)
        
        # Concatenate directly. Gradients will only flow to the local features 
        # during the forward pass matrix multiplication (hidden1 @ hidden2_all.T).
        return torch.cat(gathered_features, dim=0)

    def forward(self, zis, zjs, norm=True):

        if self.use_distributed > 1:
            # Get (normalized) hidden1 and hidden2.
            if norm:
                zis = F.normalize(zis, p=2, dim=1)
                zjs = F.normalize(zjs, p=2, dim=1)
                
            hidden1, hidden2 = zis, zjs
            hidden1_all = self.gather_features(hidden1)
            hidden2_all = self.gather_features(hidden2)
            batch_size = hidden1.shape[0]           # Local batch size
            global_batch_size = hidden1_all.shape[0] # Global batch size (if DDP)

            # Labels need to map local samples to their position in global batch
            rank = dist.get_rank()
            
            # The indices of local samples in the global batch
            labels_idx = torch.arange(start=0, end=batch_size, dtype=torch.int64).to(self.device) + rank * batch_size
            labels = F.one_hot(labels_idx, num_classes=global_batch_size).float()

            # logits: Local samples (rows) vs Global samples (cols)
            # Shape: (batch_size, global_batch_size)
            logits_ab = torch.matmul(hidden1, hidden2_all.T) / self.temperature
            logits_ba = torch.matmul(hidden2, hidden1_all.T) / self.temperature
            
            loss_a = self.softXEnt(labels, logits_ab)
            loss_b = self.softXEnt(labels, logits_ba)
            loss = (self.alpha_weight*loss_a + (1-self.alpha_weight)*loss_b)
        else:
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
