from torch import nn
from torch.nn import functional as F
import torch

class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.padding_idx = padding_idx
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        n_class = self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (n_class - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return -(true_dist * self.log_softmax(x)).sum(dim=-1).mean()

class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args['base']['alpha']
        self.beta = args['base']['beta']
        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.delta = args['base'].get('delta', 1.0)  # 新增一致性损失的权重
        self.CE_Fn = nn.CrossEntropyLoss()
        self.MSE_Fn = nn.MSELoss()


    def forward(self, out, label):
        # 各种损失的计算
        l_cc = self.MSE_Fn(out.get('w', torch.zeros_like(label['completeness_labels'])), label['completeness_labels'])
        # 检查 input 和 target 的形状是否一致

        # 计算判别损失
        l_adv = self.CE_Fn(
            out.get('effectiveness_discriminator_out', torch.zeros_like(label['effectiveness_labels'])),
            label['effectiveness_labels']
        )
        l_rec = self.MSE_Fn(out.get('rec_feats', torch.zeros_like(out.get('complete_feats', torch.empty(1)))),
                            out.get('complete_feats', torch.zeros_like(out.get('rec_feats', torch.empty(1)))))
        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels'])

        # 跨模态一致性损失
        l_consistency = out.get('consistency_loss', torch.tensor(0.0))

        # 总损失
        loss = (self.alpha * l_cc + self.beta * l_adv + self.gamma * l_rec +
                self.sigma * l_sp + self.delta * l_consistency)

        return {
            'loss': loss,
            'l_sp': l_sp,
            'l_cc': l_cc,
            'l_adv': l_adv,
            'l_rec': l_rec,
            'l_consistency': l_consistency  # 返回一致性损失
        }
