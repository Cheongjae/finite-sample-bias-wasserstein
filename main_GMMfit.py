import os
import datetime
import pickle
import argparse
from os.path import join, splitext
import setproctitle
from train_GMMfit import run_experiment 

def train(args):
    debiasstr=''
    initstr=''
    fixdata=''
    lossstr=''
    regstr=''
    if args.dist == 'Wasserstein':
        lossstr='W'
    elif args.dist == 'Wasserstein':
        lossstr='SW'
    if args.debias:
        debiasstr = '_debias'
    if args.use_init:
        initstr = '_kmeansinit'
    if args.fix_data:
        fixdata = '_fixdata'
    if args.reg > 0:
        regstr=f'_reg{args.reg}'
    savefolder = f'{lossstr}_D{args.dim}K{args.K}_mb{args.mb_size}_ds{args.dataseed}{debiasstr}{regstr}_pifixed{fixdata}{initstr}_lr{args.lr}_T{args.Nsteps}_S{args.seed}'

    os.makedirs(join('./results', savefolder), exist_ok=True)  # will not raise an error if it already exists

    res = run_experiment(args)
    
    filename = join('./results', savefolder, 'results.pickle')
    print(f'results.pickle saved in {savefolder}.')
    with open(filename, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training ')
    
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='choose gpu number')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--dataseed',
                        type=int,
                        default=42,
                        help='random seed (default: 42)')
    parser.add_argument("--seeds", nargs='*', type=int, default=None)
    parser.add_argument('--K',
                        type=int,
                        default=4,
                        help='the number of mixtures')
    parser.add_argument('--dim',
                        type=int,
                        default=2,
                        help='dimensionality of data')
    parser.add_argument("--pi", nargs='*', type=float, default=None,
                        help='mixture weights')
    parser.add_argument('--mean_constant',
                        type=float,
                        default=3.0,
                        help='a constant to set mean')
    parser.add_argument('--n_samples',
                        type=int,
                        default=10000,
                        help='the number of samples')
    parser.add_argument('--mb_size',
                        type=int,
                        default=32,
                        help='the batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='the learning rate')
    parser.add_argument('--Nsteps',
                        type=int,
                        default=20000,
                        help='the number of iterations')
    parser.add_argument('--dist',
                        type=str,
                        default='Wasserstein',
                        help='Wasserstein/Sliced-Wasserstein')
    parser.add_argument('--reg',
                        type=float,
                        default=0.0,
                        help='the entropic regularization weight')
    parser.add_argument('--scaling',
                        type=float,
                        default=0.9,
                        help='scaling parameter in geomloss')
    parser.add_argument("--debias", action='store_true', help="use bias correction")
    parser.add_argument("--fix_data", action='store_true', help="use a fixed training data")
    parser.add_argument("--use_init", action='store_true', help="use kmeans initialization")
    
    args = parser.parse_args()
    
    # time
    time_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    setproctitle.setproctitle(time_now)
    print(time_now)
    
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    if args.gpu == -1:
        args.dev = 'cpu'
    else:
        args.dev = 'cuda:' + str(args.gpu)
    
    # set random seed to reproduce the work
    if args.seeds is None:
        train(args)
    else:
        print(args.seeds)
        for seed in args.seeds:
            args.seed = seed
            train(args)
