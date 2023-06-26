from torch import nn
import torch
import torch.nn.functional as F
def rewighting_loss(loss_func, logits, label, weight):
    # loss_func = nn.CrossEntrophyLoss(reduction = "none")
    # weight :(batch, label_num)
    batch_size = label.size()[0]
    unweight_loss = loss_func(logits, label) #(batch)
    
    row_index = torch.Tensor([i for i in range(batch_size)]).long()
    column_index = torch.Tensor(label.detach().cpu().numpy()).long()

    label_weight = 1 - weight[row_index, column_index]#(batch)
    # 归一化，均值为1
    average = torch.sum(label_weight)/batch_size 
    label_weight_minmax = label_weight/average

    return torch.sum(label_weight_minmax * unweight_loss)/label.size()[0]



class JSD():
    def __init__(self,reduction='batchmean'):
        super(JSD, self).__init__()
        self.reduction = reduction

    def cal(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs= F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction=self.reduction) 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction=self.reduction) 
     
        return (0.5 * loss)
    
    

def cal_loss(debias_logits, target, bias_logits):
    #debias_logits(batch, class) torch.Tensor
    #target(batch) torch.Tensor
    #bias_probs(batch, class) torch.Tensor
    bias_probs = F.log_softmax(bias_logits, dim = -1)
    var = torch.var(bias_probs, dim = -1) #(batch) 方差
    var_max = var.max(0).values
    var_min = var.min(0).values
    var = (var-var_min)/(var_max-var_min)
    
    logprobs = F.log_softmax(debias_logits, dim=-1)
    nll_loss = - logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1) #(batch) 对应标签的负值
    smoothing_loss = torch.sum(logprobs, dim = -1) #(batch) 合并
    v_1 = 1 - 3/2 * var
    v_2 = var/2

    loss = v_1 * nll_loss - v_2 * smoothing_loss

    return loss.mean()

def cal_var(probs_bias):
    var = torch.var(probs_bias, dim = -1) #(batch) 方差
    var_max = var.max(0).values
    var_min = var.min(0).values
    var_probs = (var-var_min)/(var_max-var_min)
    
    return var_probs.tolist()