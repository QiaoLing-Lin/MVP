import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace

class myDensenet(nn.Module):
    def __init__(self, densenet):
        super(myDensenet, self).__init__()
        self.densenet = densenet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)
        # set_trace()

        x = self.densenet.features(x)

        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0)
        
        return fc, att

