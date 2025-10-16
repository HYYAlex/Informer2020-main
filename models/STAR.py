import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
class STAR(nn.Module):
    def __init__(self, d_model, d_core):
        super(STAR, self).__init__()
        self.d_model = d_model
        self.d_core = d_core

        self.gen1 = nn.Linear(d_model, d_model)
        self.gen2 = nn.Linear(d_model, d_core)
        # gen3 的输入是 d_model + d_core
        self.gen3 = nn.Linear(d_model + d_core, d_model)
        self.gen4 = nn.Linear(d_model, d_model)

    def forward(self, input, *args, **kwargs):
        # input: (B, L, D)
        B, L, D = input.shape
        combined_mean = F.gelu(self.gen1(input))      # (B,L,D)
        combined_mean = self.gen2(combined_mean)      # (B,L,d_core)

        # 简化版：直接加权平均，不做复杂采样
        # 聚合结果保持 (B,L,d_core)
        weight = F.softmax(combined_mean, dim=-1)
        combined_mean = torch.sum(combined_mean * weight, dim=-1, keepdim=True)  # (B,L,1)
        combined_mean = combined_mean.expand(-1, L, self.d_core)  # (B,L,d_core)

        # 拼接 (B,L,D_model + d_core)
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))  # (B,L,D_model)
        combined_mean_cat = self.gen4(combined_mean_cat)

        return combined_mean_cat, None


