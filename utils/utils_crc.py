import os
import time
import argparse
import torch
from torch import nn
from torch.backends import cudnn
import numpy as np
import torchvision
import torch.nn.functional as F

import logging
import copy





def confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index):
    y_pred_aug0_probas = y_pred_aug0_probas.detach()
    y_pred_aug1_probas = y_pred_aug1_probas.detach()
    y_pred_aug2_probas = y_pred_aug2_probas.detach()

    revisedY0 = part_y.clone()

    revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
    # revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1)
    revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(revisedY0.size(1), 1).transpose(0, 1)

    confidence[index, :] = revisedY0.cpu()