import argparse

def extract_args():
    # main setting
    parser = argparse.ArgumentParser(
        prog='VALEN demo file.',
        usage='Demo with partial labels.',
        description='Various algorithms of VALEN.',
        epilog='end',
        add_help=True
    )
    # optional args
    parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-2)
    parser.add_argument('-wd', help='weight decay', type=float, default=1e-4)
    parser.add_argument('-bs', help='batch size', type=int, default=256)
    parser.add_argument('-ep', help='number of epochs', type=int, default=200)
    parser.add_argument('-dt', help='type of the dataset', type=str, choices=['benchmark', 'realworld', 'uci'])
    parser.add_argument('-ds', help='specify a dataset', type=str, choices=['mnist', 'fmnist', 'kmnist', 'cifar10', 'cifar100', 'cub200',
                                                                            'FG_NET', 'lost', 'MSRCv2', 'Mirflickr', 'BirdSong',
                                                                            'malagasy', 'Soccer_Player', 'Yahoo_News', 'italian'])
    parser.add_argument('-warm_up', help='number of warm-up epochs', type=int, default=10)
    parser.add_argument('-knn', help='number of knn neighbours', type=int, default=3)
    parser.add_argument('-partial_type', help='flipping strategy', type=str, default='random', choices=['random', 'feature'])
    parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)
    parser.add_argument('-sn', help='result save file name', type=str, default='newete_kmnist3.log', required=False)
    parser.add_argument('-loss', type=str, default='valen', choices=['valen'])
    parser.add_argument('-sampling', help='the sampling times of Dirichlet', type=int, default=1, required=False)
    parser.add_argument('-z_dim', help='the dimensional of laten variate z', type=int, default=128, required=True)
    # loss paramters
    parser.add_argument('--alpha1', '-alpha1', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--alpha2', '-alpha2', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--alpha3', '-alpha3', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--beta', '-beta', type=float, default=1,help = 'balance parameter of the loss function (default=1.0)')
    # parser.add_argument('--lambda', '-lambda', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--gamma', '-gamma', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--theta', '-theta', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--sigma', '-sigma', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--correct', '-correct', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--lam1', '-lam1', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--lam2', '-lam2', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('-diri_fac', type=float, default=1000, help = 'balance parameter of the loss function (default=1.0)')
    # model args
    parser.add_argument('-gpu', type=int, default=0)
    args = parser.parse_args()
    return args

def extract_args_LE():
    # main setting
    parser = argparse.ArgumentParser(
        prog='VALEN demo file.',
        usage='Demo with partial labels.',
        description='Various algorithms of VALEN.',
        epilog='end',
        add_help=True
    )
    # optional args
    parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-2)
    parser.add_argument('-wd', help='weight decay', type=float, default=1e-4)
    parser.add_argument('-bs', help='batch size', type=int, default=256)
    parser.add_argument('-ep', help='number of epochs', type=int, default=150)
    # parser.add_argument('-dt', help='type of the dataset', type=str, choices=['benchmark', 'realworld', 'uci'])
    parser.add_argument('-ds', help='specify a dataset', type=str, choices=["Ar", "SJAFFE", "Yeast_spoem", "Yeast_spo5",
                                                                           "Yeast_dtt", "Yeast_cold", "Yeast_heat",
                                                                           "Yeast_spo", "Yeast_diau", "Yeast_elu",
                                                                           "Yeast_cdc", "Yeast_alpha", "SBU_3DFE",
                                                                           "Movie"])
    # parser.add_argument('-warm_up', help='number of warm-up epochs', type=int, default=10)
    # parser.add_argument('-knn', help='number of knn neighbours', type=int, default=3)
    # parser.add_argument('-partial_type', help='flipping strategy', type=str, default='random', choices=['random', 'feature'])
    # parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)
    # parser.add_argument('-sn', help='result save file name', type=str, default='newete_kmnist3.log', required=False)
    # parser.add_argument('-loss', type=str, default='valen', choices=['valen'])
    parser.add_argument('-sampling', help='the sampling times of Dirichlet', type=int, default=1, required=False)
    parser.add_argument('-z_dim', help='the dimensional of laten variate z', type=int, default=128, required=False)
    # loss paramters
    parser.add_argument('--alpha1', '-alpha1', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--alpha2', '-alpha2', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    # parser.add_argument('--alpha3', '-alpha3', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--beta', '-beta', type=float, default=1,help = 'balance parameter of the loss function (default=1.0)')
    # parser.add_argument('--lambda', '-lambda', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--gamma', '-gamma', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--theta', '-theta', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--sigma', '-sigma', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--correct', '-correct', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    # model args
    parser.add_argument('-gpu', type=int, default=1)
    args = parser.parse_args()
    return args