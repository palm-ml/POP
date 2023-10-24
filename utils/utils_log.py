from copy import deepcopy
import os
import time
from typing import List
import numpy as np
from torchvision.datasets.mnist import FashionMNIST

from logger import thelog


class LossLog:
    def __init__(self) -> None:
        self.loss_dict = {}

    def add_loss(self, **args):
        for key, value in args.items():
            self.loss_dict.setdefault(key, [])
            self.loss_dict[key].append(value)
    def print_loss_mean(self):
        print_str = ';'.join(map(lambda x: x[0] + ": {:.4f}".format(sum(x[1])/len(x[1])), self.loss_dict.items()))
        print(print_str)
    
    def print_loss_now(self):
        print_str = ';'.join(map(lambda x: x[0] + ": {:.4f}".format(x[1][-1]), self.loss_dict.items()))
        print(print_str)

def check_log(dir, sn, ep):
    if os.path.exists(dir + sn):
        print("exist")
        with open(dir+sn, 'r') as f:
            lines = f.readlines()
        if len(lines) >= ep:
            return False
        else:
            os.remove(dir + sn)
            return True
    else:
        return True

class TimeUse(object):
    def __init__(self, name, logger) -> None:
        super().__init__()
        self.name = name
        self.logger = logger
        
    def __enter__(self):
        self.t = time.time()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.info("Module {} : Using {} seconds.".format(self.name, time.time()-self.t))


class Monitor:
    def __init__(self, num_samples, num_classes, logger) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.alpha_matrix = np.zeros((num_samples, num_classes))
        self.temp_alpha_matrix = np.zeros((num_samples, num_classes))
        self.current_epoch = 0
        self.current_step = 0
        self.loss_log = {}
        self.current_epoch_loss = {}
        self.current_step_loss = {}
        self.logger = logger


    def monitor_alpha(self, epoch, alpha, indexes):
        if epoch == 0:
            self.alpha_matrix[indexes,:] = alpha.clone().detach().cpu().numpy()
            self.temp_alpha_matrix[indexes,:] = alpha.clone().detach().cpu().numpy()
        else:
            self.temp_alpha_matrix[indexes,:] = alpha.clone().detach().cpu().numpy()

    def print_alpha_F2(self): 
        F2 = np.linalg.norm(self.alpha_matrix - self.temp_alpha_matrix)
        self.logger.info("Current F2 value: {:.4f}".format(F2))
    
    def print_index_alpha(self, index):
        self.logger.info(self.alpha_matrix[index,:])

    def print_index_alpha_F2(self, index):
        F2 = np.linalg.norm(self.alpha_matrix[index,:] - self.temp_alpha_matrix[index,:])
        self.logger.info("Sample {} Current F2 value: {:.4f}".format(int(index), F2))

    def update_alpha(self):
        self.alpha_matrix = deepcopy(self.temp_alpha_matrix) 

    def monitor_loss(self, epoch, **args):
        for k,v in args.items():
            self.loss_log.setdefault(epoch, {})
            self.loss_log[epoch].setdefault(k, [])
            self.loss_log[epoch][k].append(v)
    
    def print_epoch_loss(self, epoch, key=None):
        if key == None:
            print_str = ""
            for k, v in self.loss_log[epoch].items():
                print_str = print_str + "loss {}: {:.4f}, ".format(k, sum(v)/len(v))
            self.logger.info(print_str)
        if key != None and type(key) == List:
            print_str = ""
            for k, v in self.loss_log[epoch].items():
                if k in key:
                    print_str = print_str + "loss {}: {:.4f}, ".format(k, sum(v)/len(v))
            self.logger.info(print_str)
        if key != None and type(key) == str:
            print_str = ""
            for k, v in self.loss_log[epoch].items():
                if k == key:
                    print_str = print_str + "loss {}: {:.4f}, ".format(k, sum(v)/len(v))
            self.logger.info(print_str)


def initLogger(args):
    id = str(time.time())
    if args.dt == 'benchmark':
        save_dir = 'results/' + str(args.loss) + '_final_results/' + '/'.join([str(args.ds), str(args.partial_type)]) + '/' + id
    if args.dt == 'realworld':
        save_dir = 'resutls/' + str(args.loss) + '_final_results/' + '/'.join(['realworld', str(args.ds)]) + '/' + id
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/result_data')
    logfilename = '_'.join([str(args.ds), str(args.partial_type)]) + '_' + id + '.log'
    logger = thelog.get_logger('expanded valen', save_dir, logfilename)
    logger.info('Parameters : ' + '\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    return logger, save_dir

def initLogger_LE(args):
    id = str(time.time())
    save_dir = 'resutls/' + 'final_results_LE/' + args.ds + '/' + id
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logfilename = id + '.log'
    logger = thelog.get_logger('expanded valen', save_dir, logfilename)
    logger.info('Parameters : ' + '\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    return logger, save_dir






if __name__ == '__main__':
    print(check_log('/data1/qiaocy/workplace/VALEN/results_benchmark_feature/mnist/', '1.log', 4))