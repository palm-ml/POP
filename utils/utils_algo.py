import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import os
import hashlib 
import errno

from torch.optim.lr_scheduler import LambdaLR


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred
    
def binarize_class(y):  
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto') 
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)     
    label = torch.from_numpy(label)
    return label


def partialize(y, t="binomial", p=0.5):
    y0 = torch.argmax(y, axis=1)
    new_y = y.clone()
    n, c = y.shape[0], y.shape[1]
    avgC = 0

    if t=='binomial':
        for i in range(n):
            row = new_y[i, :] 
            row[np.where(np.random.binomial(1, p, c)==1)] = 1
            while torch.sum(row) == 1:
                row[np.random.randint(0, c)] = 1
            avgC += torch.sum(row)
            

    if t=='pair':
        P = np.eye(c)
        for idx in range(0, c-1):
            P[idx, idx], P[idx, idx+1] = 1, p
        P[c-1, c-1], P[c-1, 0] = 1, p
        for i in range(n):
            row = new_y[i, :]
            idx = y0[i] 
            row[np.where(np.random.binomial(1, P[idx, :], c)==1)] = 1
            avgC += torch.sum(row)

    avgC = avgC / n    
    return new_y, avgC


def partialize2(y, y0, t="binomial", p=0.5):
    new_y = y.clone()
    n, c = y.shape[0], y.shape[1]
    avgC = 0

    if t=='binomial':
        for i in range(n):
            row = new_y[i, :] 
            row[np.where(np.random.binomial(1, p, c)==1)] = 1
            while torch.sum(row) == 1:
                row[np.random.randint(0, c)] = 1
            avgC += torch.sum(row)
            

    if t=='pair':
        P = np.eye(c)
        for idx in range(0, c-1):
            P[idx, idx], P[idx, idx+1] = 1, p
        P[c-1, c-1], P[c-1, 0] = 1, p
        for i in range(n):
            row = new_y[i, :]
            idx = y0[i] 
            row[np.where(np.random.binomial(1, P[idx, :], c)==1)] = 1
            avgC += torch.sum(row)

    avgC = avgC / n    
    return new_y, avgC


def check_integrity(fpath, md5): 
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): 
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    import urllib.request

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath) 

def to_logits(y):
    '''
        将y中每一行值最大的位置赋值为1, 其余位置为0
    '''
    y_ = torch.zeros_like(y)
    col = torch.argmax(y, axis=1)
    row = [ i for i in range(0, len(y))]
    y_[row, col] = 1

    return y_


def feature_partialize(train_X, train_Y, model, weight_path, device, rate=0.4):
    with torch.no_grad():
        model = model.to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        avg_C = 0
        train_X, train_Y = train_X.to(device), train_Y.to(device)
        _, outputs = model(train_X)
        train_p_Y = train_Y.clone().detach()
        partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
        partial_rate_array[torch.where(train_Y==1)] = 0
        partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
        partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
        partial_rate_array[partial_rate_array > 1.0] = 1.0
        m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
        z = m.sample()
        train_p_Y[torch.where(z == 1)] = 1.0
    avg_C = torch.sum(train_p_Y) / train_p_Y.size(0)
    return train_p_Y, avg_C.item()

def feature_partialize2(train_X, train_Y, model, weight_path, device, rate=0.4, batch_size=2000):
    with torch.no_grad():
        model = model.to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        avg_C = 0
        train_X, train_Y = train_X.to(device), train_Y.to(device)
        train_p_Y_list = []
        step = train_X.size(0) // batch_size
        for i in range(0, step):
            _, outputs = model(train_X[i*batch_size:(i+1)*batch_size])
            train_p_Y = train_Y[i*batch_size:(i+1)*batch_size].clone().detach()
            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            partial_rate_array[torch.where(train_Y[i*batch_size:(i+1)*batch_size]==1)] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0
            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()
            train_p_Y[torch.where(z == 1)] = 1.0
            train_p_Y_list.append(train_p_Y)
        train_p_Y = torch.cat(train_p_Y_list, dim=0)
        assert train_p_Y.shape[0] == train_X.shape[0]
    avg_C = torch.sum(train_p_Y) / train_p_Y.size(0)
    return train_p_Y, avg_C.item()


def confidence_update(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        _, batch_outputs = model(batchX)
        temp_un_conf = F.softmax(batch_outputs, dim=1)
        confidence[batch_index, :] = temp_un_conf * batchY # un_confidence stores the weight of each example
        #weight[batch_index] = 1.0/confidence[batch_index, :].sum(dim=1)
        base_value = confidence.sum(dim=1).unsqueeze(1).repeat(1, confidence.shape[1])
        confidence = confidence/base_value
    return confidence


def confidence_update_lw(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        device = batchX.device
        _, batch_outputs = model(batchX)
        sm_outputs = F.softmax(batch_outputs, dim=1)

        onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
        onezero[batchY > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(device)
        counter_onezero = counter_onezero.to(device)

        new_weight1 = sm_outputs * onezero
        new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight2 = sm_outputs * counter_onezero
        new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight = new_weight1 + new_weight2

        confidence[batch_index, :] = new_weight
        return confidence

def confidence_update_cavl(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        _, batch_outputs = model(batchX)
        cav = (batch_outputs * torch.abs(1 - batch_outputs)) * batchY
        cav_pred = torch.max(cav, dim=1)[1]
        gt_label = F.one_hot(cav_pred, batchY.shape[1])  # label_smoothing() could be used to further improve the performance for some datasets
        confidence[batch_index, :] = gt_label.float()

    return confidence


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.ep)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)