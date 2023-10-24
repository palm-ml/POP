from copy import deepcopy

import torch
import torchvision

from models.convnet import convnet
from models.lenet import LeNet
from models.linear import linear
from models.mlp import mlp_feature, mlp_phi
from models.VGAE import VAE_Bernulli_Decoder, Decoder_L, CONV_Decoder, CONV_Encoder_MNIST, CONV_Decoder_MNIST, \
    Z_Encoder, X_Decoder
from models.resnet import resnet
from models.VGAE import CONV_Encoder
from models.resnet34 import Resnet34
from models.resnets import resnet50
from models.wideresnet import WideResNet
from partial_models.linear_mlp_models import linear_model, mlp_model


def create_model(config, **args):
    if config.dt == "benchmark":
        if config.ds in ['mnist', 'kmnist', 'fmnist']:
            if config.partial_type == "random":
                net = mlp_feature(args['num_features'], args['num_features'], args['num_classes'])
            if config.partial_type == "feature":
                # net = mlp_phi(args['num_features'], args['num_classes'])
                net = mlp_feature(args['num_features'], args['num_features'], args['num_classes'])
        if config.ds in ['cifar10']:
            net = resnet(depth=32, n_outputs=args['num_classes'])
        if config.ds in ['cifar100']:
            net = convnet(input_channel=3, n_outputs=args['num_classes'], dropout_rate=0.25)
        if config.ds in ['cub200']:
            net = Resnet34(200)
        enc_d = deepcopy(net)
        # for cifar
        if config.ds in ['cub200']:
            enc_z = CONV_Encoder(in_channels=3,
                                 feature_dim=256,
                                 num_classes=args['num_classes'],
                                 hidden_dims=[32, 64, 128, 256, 512, 1024, 2048],
                                 z_dim=config.z_dim)
            dec_phi = CONV_Decoder(num_classes=args['num_classes'],
                                   hidden_dims=[2048, 1024, 512, 256, 128, 64, 32],
                                   z_dim=config.z_dim)
        if config.ds in ['cifar10', 'cifar100']:
            enc_z = CONV_Encoder(in_channels=3,
                                 feature_dim=32,
                                 num_classes=args['num_classes'],
                                 hidden_dims=[32, 64, 128, 256],
                                 z_dim=config.z_dim)
            # dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
            dec_phi = CONV_Decoder(num_classes=args['num_classes'],
                                   hidden_dims=[256, 128, 64, 32],
                                   z_dim=config.z_dim)
        if config.ds in ['mnist', 'kmnist', 'fmnist']:
            enc_z = CONV_Encoder_MNIST(in_channels=1,
                                       feature_dim=28,
                                       num_classes=args['num_classes'],
                                       hidden_dims=[32, 64, 128, 256],
                                       z_dim=config.z_dim)
            # dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
            dec_phi = CONV_Decoder_MNIST(num_classes=args['num_classes'],
                                         hidden_dims=[256, 128, 64, 32],
                                         z_dim=config.z_dim)
        dec_L = Decoder_L(num_classes=args['num_classes'], hidden_dim=128)
        return net, enc_d, enc_z, dec_L, dec_phi
    if config.dt == "realworld":
        net = linear(args['num_features'], args['num_classes'])
        enc_d = deepcopy(net)
        enc_z = Z_Encoder(feature_dim=args['num_features'],
                          num_classes=args['num_classes'],
                          num_hidden_layers=2,
                          hidden_size=25,
                          # z_dim=int(args['num_features'] / 10)
                          z_dim=int(args['num_classes'] * 1.5)
                          )
        dec_phi = X_Decoder(feature_dim=args['num_features'],
                            num_classes=args['num_classes'],
                            num_hidden_layers=2,
                            hidden_size=25,
                            # z_dim=int(args['num_features'] / 10),
                            z_dim = int(args['num_classes'] * 1.5)
        )
        dec_L = Decoder_L(num_classes=args['num_classes'], hidden_dim=128)
        return net, enc_d, enc_z, dec_L, dec_phi


def create_model_for_baseline(config, **args):
    if config.ds == 'cub200':
        net = Resnet34(200)
        return net
    if config.ds == 'cifar10':
        net = WideResNet(28, args['num_classes'], widen_factor=2, dropRate=0.0)
    if config.ds in ['mnist', 'kmnist', 'fmnist']:
        net = LeNet(out_dim=args['num_classes'], in_channel=1, img_sz=28) # for data augmentation
    if config.ds in ['cifar100']:
        net = WideResNet(28, args['num_classes'], widen_factor=2, dropRate=0.0)
    if config.dt == 'realworld':
        net = linear_model(input_dim=args['num_features'], output_dim=args['num_classes'])
    return net


def create_model_for_abs(config, **args):
    if config.mo == 'mlp':
        model = mlp_model(input_dim=args['num_features'], output_dim=args['num_classes'])
    elif config.mo == 'linear':
        model = linear_model(input_dim=args['num_features'], output_dim=args['num_classes'])
    elif config.mo == 'resnet':
        model = resnet(depth=32, num_classes=args['num_classes'])
    elif config.mo == 'convnet':
        model = convnet(input_channel=3, n_outputs=args['num_classes'], dropout_rate=0.25)
    return model

def create_model_LE(config, **args):
    enc_d = mlp_feature(args['num_features'], args['num_features'], args['num_classes'])
    net = deepcopy(enc_d)
    enc_z = Z_Encoder(feature_dim=args['num_features'],
                      num_classes=args['num_classes'],
                      num_hidden_layers=1,
                      hidden_size=int(args['num_classes'] * 2),
                      # z_dim=int(args['num_features'] / 10)
                      z_dim=int(args['num_classes'] * 1.5)
                      )
    dec_phi = X_Decoder(feature_dim=args['num_features'],
                        num_classes=args['num_classes'],
                        num_hidden_layers=1,
                        hidden_size=int(args['num_classes'] * 2),
                        # z_dim=int(args['num_features'] / 10),
                        z_dim=int(args['num_classes'] * 1.5)
                        )
    dec_L = Decoder_L(num_classes=args['num_classes'], hidden_dim=int(args['num_classes'] * 1.5))
    return net, enc_d, enc_z, dec_L, dec_phi


def create_model_DA(config, **args):
    if config.dt == "benchmark":
        if config.ds in ['mnist', 'kmnist', 'fmnist']:
            net = LeNet(out_dim=args['num_classes'], in_channel=1, img_sz=28)  # for data augmentation
            enc_d = deepcopy(net)
        if config.ds in ['cifar10']:
            # net = resnet(depth=32, n_outputs=args['num_classes'])
            net = WideResNet(28, args['num_classes'], widen_factor=2, dropRate=0.0)
            enc_d = WideResNet(28, args['num_classes'], widen_factor=2, dropRate=0.0)
        if config.ds in ['cifar100']:
            # net = convnet(input_channel=3, n_outputs=args['num_classes'], dropout_rate=0.25)
            net = WideResNet(28, args['num_classes'], widen_factor=2, dropRate=0.0)
            enc_d = WideResNet(28, args['num_classes'], widen_factor=2, dropRate=0.0)
        if config.ds in ['cub200']:
            net = Resnet34(200)
            enc_d = Resnet34(200)
        # enc_d = deepcopy(net)
        # for cifar
        if config.ds in ['cub200']:
            enc_z = CONV_Encoder(in_channels=3,
                                 feature_dim=256,
                                 num_classes=args['num_classes'],
                                 hidden_dims=[32, 64, 128, 256, 512, 1024, 2048],
                                 z_dim=config.z_dim)
            dec_phi = CONV_Decoder(num_classes=args['num_classes'],
                                   hidden_dims=[2048, 1024, 512, 256, 128, 64, 32],
                                   z_dim=config.z_dim)
        if config.ds in ['cifar10', 'cifar100']:
            enc_z = CONV_Encoder(in_channels=3,
                                 feature_dim=32,
                                 num_classes=args['num_classes'],
                                 hidden_dims=[32, 64, 128, 256],
                                 z_dim=config.z_dim)
            # dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
            dec_phi = CONV_Decoder(num_classes=args['num_classes'],
                                   hidden_dims=[256, 128, 64, 32],
                                   z_dim=config.z_dim)
        if config.ds in ['mnist', 'kmnist', 'fmnist']:
            enc_z = CONV_Encoder_MNIST(in_channels=1,
                                       feature_dim=28,
                                       num_classes=args['num_classes'],
                                       hidden_dims=[32, 64, 128, 256],
                                       z_dim=config.z_dim)
            # dec = VAE_Bernulli_Decoder(args['num_classes'], args['num_features'], args['num_features'])
            dec_phi = CONV_Decoder_MNIST(num_classes=args['num_classes'],
                                         hidden_dims=[256, 128, 64, 32],
                                         z_dim=config.z_dim)
        dec_L = Decoder_L(num_classes=args['num_classes'], hidden_dim=128)
        return net, enc_d, enc_z, dec_L, dec_phi