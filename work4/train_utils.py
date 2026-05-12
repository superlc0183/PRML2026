import torch
import torch.nn as nn

# ==========================================
# 1. 动态学习率优化器 (对应论文 5.3 节)
# ==========================================
class NoamOpt:
    """
    实现论文中的自定义学习率调度机制：
    先线性预热 (Warmup)，然后按步数的反平方根衰减 (Decay)
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "更新参数并更新学习率"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        "计算当前步数的学习率"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

# ==========================================
# 2. 标签平滑损失函数 (对应论文 5.4 节)
# ==========================================
class LabelSmoothing(nn.Module):
    """
    实现标签平滑：不让模型变得过度自信 (强行分配一部分概率给其他词)，
    以此来提高模型的泛化能力 (防止过拟合/死记硬背)。
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # 使用 KL 散度作为基础损失函数
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 把被平滑掉的概率均分给除了正确答案以外的所有词汇
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 给正确答案赋予 confidence (例如 0.9)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 忽略 padding (空白填充符)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())