import math

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable


def d_loss_LE(output1, target, true, eps=1e-12):
    output = F.softmax(output1, dim=1)
    l = target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    return loss, output

def partial_loss(output1, target, true, eps=1e-12):
    output = F.softmax(output1, dim=1)
    l = target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    # revisedY = revisedY * (output.clone().detach())
    revisedY = revisedY * output
    revisedY = revisedY / (revisedY + eps).sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    new_target = revisedY

    return loss, new_target

def proden_loss(output1, target, true, eps=1e-12):
    output = F.softmax(output1, dim=1)
    l = target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    revisedY = target.clone()
    revisedY[revisedY > 0] = 1
    # revisedY = revisedY * (output.clone().detach())
    revisedY = revisedY * output
    revisedY = revisedY / (revisedY).sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)
    new_target = revisedY

    return loss, new_target

def out_d_loss(output, d, target, eps=1e-12):
    revisedY = target.clone()
    revisedY[revisedY > 0] = 1
    cur_d = F.softmax(d, dim=1)
    cur_d = -torch.log(cur_d)
    cur_out = F.softmax(output, dim=1)
    revisedY = cur_out * revisedY
    revisedY = revisedY / (revisedY + eps).sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    o_d_loss = (cur_d * revisedY).sum() / cur_d.size(0)
    return o_d_loss

def out_d_loss_DA(target, d, d1, d2, consistency_criterion):
    d = F.softmax(d, dim=1)
    d1 = F.softmax(d1, dim=1)
    d2 = F.softmax(d2, dim=1)
    consist_loss0 = consistency_criterion(d, target)
    consist_loss1 = consistency_criterion(d1, target)
    consist_loss2 = consistency_criterion(d2, target)
    o_d_loss = (consist_loss0 + consist_loss1 + consist_loss2) / 3
    return o_d_loss

def out_cons_loss_DA(target, output, output1, output2, consistency_criterion):
    output = F.softmax(output, dim=1)
    output1 = F.softmax(output1, dim=1)
    output2 = F.softmax(output2, dim=1)
    consist_loss0 = consistency_criterion(output, target)
    consist_loss1 = consistency_criterion(output1, target)
    consist_loss2 = consistency_criterion(output2, target)
    loss = (consist_loss0 + consist_loss1 + consist_loss2) / 3
    return loss

def out_d_loss_LE(output, d, eps=1e-12):
    output = F.sigmoid(output)
    cur_d = F.sigmoid(d)
    cur_d = -torch.log(cur_d)
    # revisedY = cur_out * revisedY
    # revisedY = revisedY / (revisedY + eps).sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    o_d_loss = (cur_d * output).sum() / cur_d.size(0)
    return o_d_loss, output

def revised_target(output, target):
    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    # revisedY = revisedY * (output.clone().detach())
    revisedY = revisedY * output
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    new_target = revisedY

    return new_target

def threshold_loss(output, target, threshold):
    output1 = F.sigmoid(output)
    output2 = F.softmax(output, dim=1)
    revisedY1 = target.clone()
    label = target.clone()
    label[label>0] = 1
    values, indices = (output1.clone().detach()*label).topk(k=2, dim=1)
    delta_values = values[:, 0] - values[:, 1]
    corrected_labels = torch.zeros_like(output1)
    row_indexes = [i for i in range(0, output1.size(0))]
    col_indexes = indices[:,0]
    corrected_labels[row_indexes, col_indexes] = 1.0
    # 找出小于threshold的样本，将其标签修正
    revisedY1[delta_values>threshold] = corrected_labels[delta_values>threshold] + 0.0
    l = revisedY1 * torch.log(output2)
    loss = (-torch.sum(l)) / l.size(0)
    # loss = F.binary_cross_entropy(output1, revisedY1)
    # l = revisedY1 * torch.log(output1) + label * (1 - revisedY1) * torch.log(1 - output1)
    # loss = (-torch.sum(l)) / l.size(0)

    # output2 = F.softmax(output, dim=1)
    # revisedY = target.clone()
    # revisedY[revisedY > 0]  = 1
    # revisedY = revisedY * output2
    # revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    revisedY = target.clone()
    revisedY[revisedY > 0]  = 1
    revisedY = revisedY * (output2.clone().detach())
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)
    # revisedY = F.sigmoid(revisedY)
    # revisedY = revisedY / torch.max(revisedY, dim=1, keepdim=True)
    # 添加修正后的标签
    row_indexes, col_indexes = torch.where(revisedY1 == 1)
    corrected_num = row_indexes.size(0)
    # revisedY[row_indexes] = revisedY[row_indexes] * 0.0
    # revisedY[row_indexes, col_indexes] = 1.0 
    new_target = revisedY

    return loss, new_target, corrected_num


def weighted_ce_loss(y_, y, w):
    return -torch.sum(y * torch.log(F.softmax(y_*w, dim=1)))/y.size(0)


def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss


def rc_loss(outputs, confidence, index):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = - ((final_outputs).sum(dim=1)).mean()
    return average_loss

def lws_loss(outputs, partialY, confidence, index, lw_weight, lw_weight0, epoch_ratio):
    device = outputs.device
    onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    onezero[partialY > 0] = 1
    counter_onezero = 1 - onezero
    onezero = onezero.to(device)
    counter_onezero = counter_onezero.to(device)

    sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss1 = sig_loss1.to(device)
    sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
    sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (
        1 + torch.exp(-outputs[outputs > 0]))
    l1 = confidence[index, :] * onezero * sig_loss1
    average_loss1 = torch.sum(l1) / l1.size(0)

    sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss2 = sig_loss2.to(device)
    sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
    sig_loss2[outputs < 0] = torch.exp(
        outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
    l2 = confidence[index, :] * counter_onezero * sig_loss2
    average_loss2 = torch.sum(l2) / l2.size(0)

    average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
    return average_loss, lw_weight0 * average_loss1, lw_weight * average_loss2

def min_loss(output1, target, eps=1e-12):
    output1 = F.softmax(output1, dim=1)
    l =  - target * torch.log(output1 + eps)
    new_labels = torch.zeros_like(output1)
    row_indexes = [ i for i in range(0, output1.size(0))]
    l_clone = l.clone().detach()
    l_clone[l_clone==0] = l_clone.max()
    col_indexes = torch.argmin(l_clone, dim=1)
    new_labels[row_indexes, col_indexes] = 1
    return torch.sum(l * new_labels)/l.size(0)


def gauss_kl_loss(mu, logvar, eps = 1e-12):
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def dirichlet_kl_loss(alpha, prior_alpha):
    KLD = torch.mvlgamma(alpha.sum(1), p=1)-torch.mvlgamma(alpha, p=1).sum(1)-torch.mvlgamma(prior_alpha.sum(1), p=1)+torch.mvlgamma(prior_alpha, p=1).sum(1)+((alpha-prior_alpha)*(torch.digamma(alpha)-torch.digamma(alpha.sum(dim=1, keepdim=True).expand_as(alpha)))).sum(1)
    return KLD.mean()


# def kl_loss(output, d, method=None):
#     output = F.softmax(output, dim=1)
#     d = F.softmax(d, dim=1)
#     if method == 'right':
#         loss = output*torch.log(output) - output*torch.log(d)
#     if method == 'left':
#         loss = d * torch.log(d) - d * torch.log(output)
#     if method == None:
#         right = output*torch.log(output) - output*torch.log(d)
#         left = d * torch.log(d) - d * torch.log(output)
#         loss = right + left
#     return loss.mean()

def kl_loss(output, d, target, eps=1e-8):
    output = F.softmax(output, dim=1)
    right_weight = output.clone().detach() * target
    right_weight = right_weight / right_weight.sum(dim=1, keepdim=True)
    right = torch.zeros_like(output)
    left = torch.zeros_like(output)
    right[target == 1] = ( - right_weight.clone().detach() * torch.log(d + eps))[target == 1]
    left[target == 1] =  ( - d.clone().detach() * torch.log(output+eps))[target == 1]
    print("output",output[0])
    print("d",d[0])
    print("ri-w", right_weight[0])
    print("ri", right[0])
    print("target", target[0])
    print(right.sum().item())
    print(left.sum().item())
    loss = right + left
    loss = loss.sum() / loss.size(0)
    return loss

def label_loss(d, labels, eps=1e-12):
    d = F.softmax(d, dim=1)
    l = labels * torch.log(d)
    loss = (-torch.sum(l)) / l.size(0)
    return loss


def alpha_loss(alpha, prior_alpha):
    KLD = torch.mvlgamma(alpha.sum(1), p=1) - torch.mvlgamma(alpha, p=1).sum(1) \
          - torch.mvlgamma(prior_alpha.sum(1), p=1) + torch.mvlgamma(prior_alpha, p=1).sum(1) + \
          ((alpha - prior_alpha) *
           (torch.digamma(alpha) - torch.digamma(alpha.sum(dim=1, keepdim=True).expand_as(alpha)))).sum(1)
    return KLD.mean()


def BetaMAP_loss(output, target, alpha, beta):
    L1 = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    L1 = (-torch.sum(L1))/L1.size(0)
    L2 = alpha * torch.log(output) + beta * torch.log(1-output)
    L2 = (-torch.sum(L2))/L2.size(0)
    L = 0.01 * L1 + 0.99 * L2
    print(L1.item(), L2.item(), L.item())
    return L


def MAP_loss(t_o, c_o, r_t, r_c, targets, alpha, beta, gamma):
    # 从候选集合中随机划分真实标签和候选标签
    L1_1 = torch.log(t_o) * r_t
    # L1_2 = torch.log(c_o) * r_c + torch.log(1 - c_o) * (1 - r_c)
    L1_1 = (- torch.sum(L1_1))/L1_1.size(0)
    # L1_2 = (- torch.mean(L1_2))
    # L1 = L1_1 + L1_2
    gamma = (gamma - 1) * targets
    gamma = gamma / torch.sum(gamma, keepdim=True, dim=1)
    L2 = gamma  * torch.log(t_o)
    L2 = (- torch.sum(L2))/L2.size(0)
    # print(t_o[0:2])
    # L3 = alpha*torch.log(c_o) + beta*torch.log(1-c_o)
    # L3 = - torch.mean(L3)
    # print(L1_1, L1_2, L2, L3)
    # if torch.isnan(L1):
    #     exit()
    # L = L1 + L2 + L3
    print(gamma[0:2])
    print(L1_1.item(), L2.item())
    L = L2
    return L


def dcnn_loss(output, target, teacher, weights, vt, alpha):
    loss_cr = _cross_entropy(output, target)
    loss_se = self_entropy(output)
    loss_STcr = ST_cross_entropy(output, teacher, weights)

    loss = loss_cr + alpha * loss_se + vt * loss_STcr

    return loss


def _cross_entropy(prediction, labels):
    _cross_entropy_singel = -torch.sum((1 - labels) * torch.log((1 - prediction) + 1e-5), dim=1)
    _cross_entropy_mean = torch.mean(_cross_entropy_singel)
    return _cross_entropy_mean


def self_entropy(prediction):
    self_entropy_singel = -torch.sum(prediction * torch.log(prediction + 1e-10), dim=1)
    self_entropy_mean = torch.mean(self_entropy_singel)
    return self_entropy_mean


def ST_cross_entropy(prediction, teacher, weights):
    cross_entropy_singel = -weights * torch.sum((teacher * torch.log(prediction + 1e-5)), dim=1)
    cross_entropy_mean = torch.mean(cross_entropy_singel)
    return cross_entropy_mean

# my test loss
def proden_loss_(output1, target, true, eps=1e-1):
    revisedY = target.clone()
    revisedY[revisedY > 0] = 1
    output = F.softmax(output1, dim=1)
    arg_max = torch.argmax(output, dim=1, keepdim=True)
    num_Y = torch.sum(revisedY, dim=1, keepdim=True).repeat(1, revisedY.size(1))
    pre_y = torch.zeros_like(target).scatter(1, arg_max, 1)
    weight = ((1 - eps) * pre_y + eps * (revisedY - pre_y) / num_Y).detach()
    l = weight * target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    # revisedY = target.clone()
    # revisedY[revisedY > 0] = 1
    # revisedY = revisedY * (output.clone().detach())
    revisedY = revisedY * output
    revisedY = revisedY / (revisedY).sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)
    new_target = revisedY
    return loss, new_target


def mae_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.L1Loss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, partialY.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss


def mse_loss(outputs, Y):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.MSELoss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, Y.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss


def gce_loss(outputs, Y):
    q = 0.7
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1 - (pow_outputs * Y).sum(dim=1)) / q  # n
    return sample_loss


def phuber_ce_loss(outputs, Y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trunc_point = 0.1
    n = Y.shape[0]
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * Y
    final_confidence = final_outputs.sum(dim=1)

    ce_index = (final_confidence > trunc_point)
    sample_loss = torch.zeros(n).to(device)

    if ce_index.sum() > 0:
        ce_outputs = outputs[ce_index, :]
        logsm = nn.LogSoftmax(dim=-1)  # because ce_outputs might have only one example
        logsm_outputs = logsm(ce_outputs)
        final_ce_outputs = logsm_outputs * Y[ce_index, :]
        sample_loss[ce_index] = - final_ce_outputs.sum(dim=-1)

    linear_index = (final_confidence <= trunc_point)

    if linear_index.sum() > 0:
        sample_loss[linear_index] = -math.log(trunc_point) + (-1 / trunc_point) * final_confidence[linear_index] + 1

    return sample_loss


def cce_loss(outputs, Y):
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    final_outputs = logsm_outputs * Y
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss


def focal_loss(outputs, Y):
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = logsm_outputs * Y * (1 - sm_outputs) ** 0.5
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss


def pll_estimator(loss_fn, outputs, partialY, device):
    n, k = partialY.shape[0], partialY.shape[1]
    comp_num = partialY.sum(dim=1)
    temp_loss = torch.zeros(n, k).to(device)

    for i in range(k):
        tempY = torch.zeros(n, k).to(device)
        tempY[:, i] = 1.0
        temp_loss[:, i] = loss_fn(outputs, tempY)

    coef = 1.0 / comp_num
    total_loss = coef * (temp_loss * partialY).sum(dim=1)
    # total_loss = total_loss.sum()
    return total_loss.mean()

def get_loss(args):
    loss_fn = None
    if args.loss == 'mae':
        loss_fn = mae_loss
    elif args.loss == 'mse':
        loss_fn = mse_loss
    elif args.loss == 'cce':
        loss_fn = cce_loss
    elif args.loss == 'gce':
        loss_fn = gce_loss
    elif args.loss == 'phuber_ce':
        loss_fn = phuber_ce_loss
    elif args.loss == 'fl':
        loss_fn = focal_loss
    return loss_fn


def loss_coteaching_pll(y_1, y_2, t, forget_rate, device):
    pll_num = torch.sum(t, dim=1)
    extended_y1 = []
    for i, num in enumerate(pll_num):
        extended_y1.append(y_1[i, :].repeat(int(num), 1))
    y_1 = torch.cat(extended_y1, dim=0)
    extended_y2 = []
    for i, num in enumerate(pll_num):
        extended_y2.append(y_2[i, :].repeat(int(num), 1))
    y_2 = torch.cat(extended_y2, dim=0)
    t = torch.nonzero(t)[:, 1]
    loss_1 = F.cross_entropy(y_1, t, reduce=False)
    ind_1_sorted = np.argsort(loss_1.cpu().data)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce=False)
    ind_2_sorted = np.argsort(loss_2.cpu().data)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    # return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember


def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, step):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id = np.zeros(labels.size(0), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1):
        if p1 != pred2[idx]:
            disagree_id.append(idx)
            logical_disagree_id[idx] = True

    temp_disagree = ind * logical_disagree_id.astype(np.int64)
    # temp_disagree = ind.dot(logical_disagree_id.astype(np.int64))
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0] == len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]

    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id]
        update_outputs2 = outputs2[disagree_id]

        loss_1, loss_2 = loss_coteaching(update_outputs, update_outputs2, update_labels,
                                                                     forget_rate, ind_disagree)
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        cross_entropy_1 = ce_loss_with_prob(update_outputs, update_labels)
        cross_entropy_2 = ce_loss_with_prob(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step * cross_entropy_1) / labels.size()[0]
        loss_2 = torch.sum(update_step * cross_entropy_2) / labels.size()[0]

    return loss_1, loss_2


def loss_coteaching(y_1, y_2, t, forget_rate, ind):
    loss_1 = ce_loss_with_prob(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = ce_loss_with_prob(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember].cpu()
    ind_2_update=ind_2_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    loss_1_update = ce_loss_with_prob(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = ce_loss_with_prob(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember


def ce_loss_with_prob(y, t, **kwargs):
    y_t = torch.log_softmax(y, dim=1)
    return -torch.sum(t * y_t, dim=1)
