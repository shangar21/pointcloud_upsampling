from emd import earth_mover_distance
import torch

class BoltLoss(torch.nn.Module):
    def __init__(self):
        super(BoltLoss, self).__init__()

    def forward(self, x, y):
        emd_loss = earth_mover_distance(x, y)
        return emd_loss


        


