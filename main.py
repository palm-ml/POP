# import
import argparse
import time
import traceback
from copy import deepcopy

import torch
import torch.nn.functional as F
from utils.data_factory import partialize, create_test_loader, \
    create_train_loader_DA, extract_data_DA
from utils.metrics import accuracy_check
from utils.model_factory import create_model_for_baseline
from utils.utils_log import TimeUse, initLogger
from utils.utils_loss import proden_loss
from utils.utils_seed import set_seed

parser = argparse.ArgumentParser(
    prog='baseline demo file.',
    usage='Demo with partial labels.',
    epilog='end',
    add_help=True
)
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=5e-2)
parser.add_argument('-wd', help='weight decay', type=float, default=1e-3)
parser.add_argument('-bs', help='batch size', type=int, default=256)
parser.add_argument('-ep', help='number of epochs', type=int, default=500)
parser.add_argument('-dt', help='type of the dataset', type=str, default='realworld',
                    choices=['benchmark', 'realworld', 'uci'])
parser.add_argument('-ds', help='specify a dataset', type=str, default='italian',
                    choices=['mnist', 'fmnist', 'kmnist', 'cifar10', 'cifar100', 'cub200',
                             'FG_NET', 'lost', 'MSRCv2', 'Mirflickr', 'BirdSong',
                             'malagasy', 'Soccer_Player', 'Yahoo!_News', 'italian'])
parser.add_argument('-partial_type', help='flipping strategy', type=str, default='feature',
                    choices=['random', 'feature'])
parser.add_argument('-loss', type=str, default='pop')
parser.add_argument('-theta', type=float, default=1e-3)
parser.add_argument('-inc', type=float, default=1e-3)
parser.add_argument('-warm_up', type=int, default=20)
parser.add_argument('-gpu', type=int, default=0)

args = parser.parse_args()
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else 'cpu')
logger, save_dir = initLogger(args)


# train benchmark
def train_benchmark(config):
    # data and model
    with TimeUse("Extract Data", logger):
        set_seed(0)
        train_X_DA, train_X, train_Y, test_X, test_Y, valid_X, valid_Y = next(extract_data_DA(config))
    print(train_X.shape)
    print(train_X_DA.shape)
    num_samples = train_X.shape[0]
    train_X_shape = train_X.shape
    train_X = train_X.view((num_samples, -1))
    num_features = train_X.shape[-1]
    train_X = train_X.view(train_X_shape)
    num_classes = train_Y.shape[-1]
    with TimeUse("Create Model", logger):
        consistency_criterion = torch.nn.KLDivLoss(reduction='batchmean').cuda()
        net = create_model_for_baseline(args, num_features=num_features, num_classes=num_classes)
        net.to(device)
    train_p_Y, avgC = partialize(config, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y,
                                 dim=num_features, device=device)
    set_seed(int(time.time()) % (2 ** 16))
    logger.info("The Training Set has {} samples and {} classes".format(num_samples, num_features))
    logger.info("Average Candidate Labels is {:.4f}".format(avgC))
    # train_loader = create_train_loader(train_X, train_Y, train_p_Y)
    train_loader = create_train_loader_DA(train_X_DA, train_Y, train_p_Y, batch_size=config.bs, ds=config.ds)
    valid_loader = create_test_loader(valid_X, valid_Y, batch_size=config.bs)
    test_loader = create_test_loader(test_X, test_Y, batch_size=config.bs)

    rollWindow = 5
    theta = args.theta
    inc = args.inc
    confidence = deepcopy(train_loader.dataset.train_p_Y)
    ori_correction_label_matrix = train_loader.dataset.train_p_Y.clone().to(device)
    pre_correction_label_matrix = train_loader.dataset.train_p_Y.clone().to(device)
    correction_label_matrix = train_loader.dataset.train_p_Y.clone().to(device)
    f_record = torch.zeros([rollWindow, num_samples, num_classes]).to(device)
    true_label_matrix = train_loader.dataset.train_Y.clone().to(device)
    confidence = confidence.to(device)
    confidence = confidence / confidence.sum(axis=1)[:, None]
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)
    best_valid = 0
    best_test = 0
    for epoch in range(0, args.ep):
        net.train()
        for features, features1, features2, targets, trues, indexes in train_loader:
            features, features1, features2, targets, trues = map(lambda x: x.to(device), (features, features1, features2, targets, trues))
            _, y_pred_aug0 = net(features)
            _, y_pred_aug1 = net(features1)
            _, y_pred_aug2 = net(features2)

            L_ce, new_labels = proden_loss(y_pred_aug0, confidence[indexes, :].clone().detach(), None)
            L_ce1, new_labels1 = proden_loss(y_pred_aug1, confidence[indexes, :].clone().detach(), None)
            L_ce2, new_labels2 = proden_loss(y_pred_aug2, confidence[indexes, :].clone().detach(), None)
            final_loss = (L_ce + L_ce1 + L_ce2) / 3
            confidence[indexes, :] = new_labels.clone().detach()

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

        f_record[epoch % rollWindow, :] = confidence
        if epoch >= args.warm_up and epoch % rollWindow == 0:
            temp_prob_matrix = f_record.mean(0)
            # label correction
            temp_prob_matrix = temp_prob_matrix / temp_prob_matrix.sum(dim=1).repeat(temp_prob_matrix.size(1),
                                                                                     1).transpose(0, 1)
            pre_correction_label_matrix = correction_label_matrix.clone()
            correction_label_matrix[temp_prob_matrix / torch.max(temp_prob_matrix, dim=1, keepdim=True)[0] < theta] = 0
            tmp_label_matrix = temp_prob_matrix * correction_label_matrix
            confidence = tmp_label_matrix / tmp_label_matrix.sum(dim=1).repeat(tmp_label_matrix.size(1), 1).transpose(0, 1)
            if theta < 0.4:
                if torch.sum(
                        torch.not_equal(pre_correction_label_matrix, correction_label_matrix)) < 0.0001 * num_samples * num_classes:
                    theta *= (inc + 1)
                    logger.info('\t\tupdate the threshold theta : ' + str(theta))


        scheduler.step()
        net.eval()
        valid_acc = accuracy_check(valid_loader, net, device)
        test_acc = accuracy_check(test_loader, net, device)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_test = test_acc
        logger.info("Epoch {}, valid acc: {:.4f}, test acc: {:.4f}".format(epoch, valid_acc, test_acc))
    logger.info('early stopping results: valid acc: {:.4f}, test acc: {:.4f}'.format(best_valid, best_test))

# enter
if __name__ == "__main__":
    try:
        if args.dt == "benchmark":
            train_benchmark(args)
    except Exception as e:
        logger.error("Error : " + str(e))
        logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
