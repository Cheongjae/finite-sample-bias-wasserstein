# Written with reference to the code available at the following repositories: https://github.com/kimiandj/slicedwass_abc, https://github.com/kilianFatras/minibatch_Wasserstein
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import time
from random import choices
from scipy import misc
import ot

from utils_GMMfit import sample_gmm, random_covariances_qr
from geomloss import SamplesLoss
import seaborn as sns; sns.set(color_codes=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        

class Neural_network(nn.Module):
    """docstring for Neural_network."""
    def __init__(self, z_dim=10, output_dim=2):
        super(Neural_network, self).__init__()
        self.linear1 = nn.Linear(z_dim, 128)
        self.linear2 = nn.Linear(128,32)
        self.linear3 = nn.Linear(32,output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class Sqrt0(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        result[input < 0] = 0
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output / (2*result)
        grad_input[result == 0] = 0
        return grad_input

def sqrt_0(x):
    return Sqrt0.apply(x)


def squared_distances(x, y):
    if x.dim() == 2:
        D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
        D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    elif x.dim() == 3:  # Batch computation
        D_xx = (x*x).sum(-1).unsqueeze(2)  # (B,N,1)
        D_xy = torch.matmul( x, y.permute(0,2,1) )  # (B,N,D) @ (B,D,M) = (B,N,M)
        D_yy = (y*y).sum(-1).unsqueeze(1)  # (B,1,M)
    else:
        print("x.shape : ", x.shape)
        raise ValueError("Incorrect number of dimensions")

    return D_xx - 2*D_xy + D_yy


def distances(x, y, useL1=False):
    if useL1:
        return sqrt_0( squared_distances(x,y) )
    else:
        return squared_distances(x,y)


def emd(mb_C, mb_size):
    mb_a, mb_b = ot.unif(mb_size), ot.unif(mb_size)
    return ot.emd(mb_a, mb_b, mb_C.detach().cpu().numpy().copy())

def sym_mat_sqrt(mat):
    mat = 0.5*(mat + mat.T)
    d, v = np.linalg.eigh(mat)
    d = np.clip(d, 0.0, None)              # guard tiny negatives
    return np.dot(v,np.dot(np.diag(d**0.5),v.T))

def wass_gaussians(mu1, mu2, Sigma1, Sigma2):
    """
    Computes the Wasserstein distance of order 2 between two Gaussian distributions
    """
    d = mu1.shape[0]
    if d == 1:
        w2 = (mu1 - mu2)**2 + (np.sqrt(Sigma1) - np.sqrt(Sigma2))**2
    else:
        Sigma2_sqrt = sym_mat_sqrt(Sigma2)
        prodSigmas = Sigma2_sqrt @ Sigma1 @ Sigma2_sqrt
        w2 = np.linalg.norm(mu1 - mu2)**2 + np.trace(Sigma1 + Sigma2 - 2*sym_mat_sqrt(prodSigmas))
    return np.sqrt(w2)

def squared_distances_np(x, y):
    # x, y are numpy arrays
    
    if x.ndim == 2:
        # x: (N, D)
        # y: (M, D)
        D_xx = np.sum(x * x, axis=-1, keepdims=True)   # (N,1)
        D_xy = np.matmul(x, y.T)                       # (N,M)
        D_yy = np.sum(y * y, axis=-1, keepdims=True).T # (1,M)

    elif x.ndim == 3:
        # x: (B, N, D)
        # y: (B, M, D)
        D_xx = np.sum(x * x, axis=-1, keepdims=True)   # (B,N,1)
        D_xy = np.matmul(x, np.transpose(y, (0,2,1)))  # (B,N,M)
        D_yy = np.sum(y * y, axis=-1, keepdims=True)   # (B,M,1)
        D_yy = np.transpose(D_yy, (0,2,1))             # (B,1,M)

    else:
        print("x.shape:", x.shape)
        raise ValueError("Incorrect number of dimensions")

    return D_xx - 2 * D_xy + D_yy

def emd_np(mb_C, mb_size):
    mb_a, mb_b = ot.unif(mb_size), ot.unif(mb_size)
    return ot.emd(mb_a, mb_b, mb_C)

def wasserstein_np(X, Y):
    mb_size = len(X)
    C = squared_distances_np(X, Y)
    pi = emd_np(C, mb_size)
    return np.sum(pi * C)
    

def sliced_wasserstein(X,
                       Y,
                       n_proj = 128,
                       p = 2,
                       seed = None, theta = None) -> torch.Tensor:
    """
    Exact (unregularized) Sliced W_p^p via random 1D projections.
    X, Y: tensors of shape (N, d) with the same N (equally-weighted samples).
    Returns the average of 1D W_p^p over 'n_proj' random directions.
    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape (N, d).")
    device = X.device
    N, d = X.shape

    # Random unit directions
    if theta is None:
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
            theta = torch.randn((n_proj, d), generator=gen, device=device)
        else:
            theta = torch.randn((n_proj, d), device=device)
        theta = theta / (theta.norm(dim=1, keepdim=True) + 1e-12)  # (n_proj, d)

    # Project, sort per projection (column), and compute exact 1D W_p^p
    Xproj = X @ theta.T                     # (N, n_proj)
    Yproj = Y @ theta.T                     # (N, n_proj)
    Xs, _ = torch.sort(Xproj, dim=0)        # (N, n_proj)
    Ys, _ = torch.sort(Yproj, dim=0)        # (N, n_proj)

    diff = (Xs - Ys).abs()
    if p == 1:
        return diff.mean(), theta
    elif p == 2:
        return (diff * diff).mean(), theta
    else:
        return diff.pow(p).mean(), theta

def sinkhorn_divergence(x, y, eps=0.05, p=2, scaling=0.9):
    device = x.device
    blur = eps**(1.0/p)
    loss = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling, debias=True, backend="tensorized")
    return loss(x, y)

def mean_cov(data):
    mean = np.mean(data, axis=0)
    X = data - mean
    cov = np.dot(X.T, X) / len(data)
    return mean, cov

def train(data, mb_size, L=1, z_dim=10, dist='Wasserstein', lr=.005, Nsteps = 10001,
                 device='cpu', alpha=0, reg_start=0, useL1=False, init=None, 
          share_theta=False, imagesdir=None, chkptdir=None, opt='Adam', data_param = None, loss_param = None) :
    """Flows along the gradient of the cost function, using a simple Euler scheme.

    Parameters:
        loss ((x_i,data) -> torch float number):
            Real-valued loss function.
        lr (float, default = .005):
            Learning rate
    """

    # Parameters for the gradient descent
    
    display_its = [int(Nsteps/8), int(Nsteps/4), int(Nsteps/2), Nsteps-1] #plot results for wanted iteration
    print(display_its)
    loss_error = []
    loss_wass = []
    loss_corr = []
    fx_set = []

    # Make sure that we won't modify the reference samples
    data = data.clone()
    data = data.to(device)
    nt = data.size()[0]
    print(data.device)

    net = Neural_network(z_dim=z_dim, output_dim = data.shape[1])
    net.to(device)
    if init is None:
        net.weight_init(mean=0.0, std=0.05)
    else:
        net.load_state_dict(init)
    if opt == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0, 0.9))
    elif opt == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise NotImplementedError

    
    t_0 = time.time()
    plt.figure(figsize=(18,6)) ; k = 1
    
    for i in range(Nsteps): # Euler scheme ===============
        optimizer.zero_grad()
        
        
        mb_x = torch.randn(mb_size, z_dim).type(torch.FloatTensor).to(device) #Batch source

        if data_param is not None:
            if data_param['type'] == 'GMM':
                prior, means, covs = data_param['prior'], data_param['means'], data_param['covs']
                mb_y, _ = sample_gmm(prior, means, covs, n_samples=mb_size)
                mb_y = torch.FloatTensor(mb_y).to(device)
            elif data_param['type'] == 'Gaussian':
                mean, cov = data_param['mean'], data_param['cov']
                mb_y = np.random.multivariate_normal(mean=mean, cov=cov, size=mb_size)
                mb_y = torch.FloatTensor(mb_y).to(device)
        else:
            mb_t = np.random.choice(nt, size=mb_size, replace=False) #Batch target
            mb_y = data[mb_t]

        f_x = net(mb_x) # G(z)
        if dist == 'Wasserstein':
            C = distances(f_x, mb_y, useL1)
            pi = torch.as_tensor(emd(C, mb_size)).to(device)
            Loss = torch.sum(pi * C.double())
            loss = Loss.item()
            bias_corr = torch.tensor(0.).to(device)
            if alpha != 0 and i >= reg_start:
                for _ in range(L):
                    mb_x1 = torch.randn(mb_size, z_dim).type(torch.FloatTensor).to(device) #Batch source
                    f_x1 = net(mb_x1) # G(z)
                    mb_x2 = torch.randn(mb_size, z_dim).type(torch.FloatTensor).to(device) #Batch source
                    f_x2 = net(mb_x2) # G(z)
                    C2 = distances(f_x1, f_x2, useL1)
                    pi2 = torch.as_tensor(emd(C2, mb_size)).to(device)
                    bias_corr += torch.sum(pi2 * C2.double())
                Loss -= alpha*bias_corr/L
        elif dist == 'Sinkhorn':
            Loss = sinkhorn_divergence(f_x, mb_y, eps=loss_param['eps'], p=2, scaling=loss_param['scaling'])
            loss = Loss.item()
            bias_corr = torch.tensor(0.).to(device)
            if alpha != 0 and i >= reg_start:
                for _ in range(L):
                    mb_x1 = torch.randn(mb_size, z_dim).type(torch.FloatTensor).to(device) #Batch source
                    f_x1 = net(mb_x1) # G(z)
                    mb_x2 = torch.randn(mb_size, z_dim).type(torch.FloatTensor).to(device) #Batch source
                    f_x2 = net(mb_x2) # G(z)
                    bias_corr += sinkhorn_divergence(f_x1, f_x2, eps=loss_param['eps'], p=2, scaling=loss_param['scaling'])
                Loss -= alpha*bias_corr/L
        elif dist == 'Sliced-Wasserstein':
            f_x = net(mb_x) # G(z)
            Loss, theta = sliced_wasserstein(f_x, mb_y)
            loss = Loss.item()
            bias_corr = torch.tensor(0.).to(device)
            if not share_theta:
                theta = None
            if alpha != 0:
                for _ in range(L):
                    mb_x1 = torch.randn(mb_size, z_dim).type(torch.FloatTensor).to(device) #Batch source
                    f_x1 = net(mb_x1) # G(z)
                    mb_x2 = torch.randn(mb_size, z_dim).type(torch.FloatTensor).to(device) #Batch source
                    f_x2 = net(mb_x2) # G(z)
                    temp, _ = sliced_wasserstein(f_x1, f_x2, theta=theta)
                    bias_corr += temp
                Loss -= alpha*bias_corr/L
        else:
            raise NotImplementedError
        Loss.backward() # Compute gradient
        optimizer.step()

        if i%max(1,Nsteps//30)==0:
            print(i)
            with torch.no_grad():
                loss_error.append(Loss.item())
                loss_wass.append(loss)
                loss_corr.append(bias_corr.item()/L)
                # save model
                torch.save(net.state_dict(), '{}/net_step_{}.pth'.format(chkptdir, i + 1))

        if i in display_its : # display
            mb_x = torch.randn(nt, z_dim).type(torch.FloatTensor).to(device)
            f_x = net(mb_x)
            arr = f_x.cpu().detach().numpy()
            fx_set.append(arr)
            try:
                ax = plt.subplot(1,4,k)
                plt.set_cmap("hsv")
                sns.kdeplot(f_x.cpu().detach().numpy(), ax=ax, fill=True)
                
                sns.kdeplot(x=arr[:,0], y=arr[:,1], ax=ax, fill=True)
                ax.set_title("MB={}, t = {:1.2f}".format(mb_size, lr*i))
                #plt.gca().set_aspect('equal', adjustable='box')
                plt.xticks([], []); plt.yticks([], [])
                k = k+1
            except ValueError as e:
                if "Contour levels must be increasing" in str(e):
                    print("⚠️ contour skipped: levels not increasing")
                else:
                    raise e
    plt.title("MB batch : t = {:1.2f}, elapsed time: {:.4f}s/it".format(lr*i, (time.time() - t_0)/Nsteps ))
    plt.tight_layout()
    plt.savefig(os.path.join(imagesdir, 'mb_GAN_2D.png'))
    plt.show()
    
    # mb_x = torch.randn(nt, z_dim).type(torch.FloatTensor).to(device)
    # f_x = net(mb_x)

    #### gather results
    results = dict()
    results['loss'] = loss_error
    results['wass'] = loss_wass
    results['bias_corr'] = loss_corr
    results['L'] = L
    results['alpha'] = alpha
    results['net'] = net
    results['fx_set'] = fx_set
    data_np = data.cpu().numpy()
    results['fx_mean_cov'] = [mean_cov(fx) for fx in fx_set]
    if dist == 'Sinkhorn':
        results['fx_wass'] = [sinkhorn_divergence(data, torch.FloatTensor(fx).to(device), eps=loss_param['eps'], p=2, scaling=loss_param['scaling']).item() for fx in fx_set]
    else:
        results['fx_wass'] = [wasserstein_np(data_np, fx) for fx in fx_set]
    
    return results

def main():
    # train args
    parser = argparse.ArgumentParser(description='Training ')
    parser.add_argument('--outdir', default='./output/', help='directory to output images and model checkpoints')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='choose gpu number.')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--dataseed',
                        type=int,
                        default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--datatype',
                        type=str,
                        default='Gaussian',
                        help='Gaussian/GMM')
    parser.add_argument("--seeds", nargs='*', type=int, default=None)
    parser.add_argument('--K',
                        type=int,
                        default=4,
                        help='the number of mixtures')
    parser.add_argument('--L',
                        type=int,
                        default=1,
                        help='the number of MC in bias correction term')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.0,
                        help='bias correction coefficient')
    parser.add_argument('--dim',
                        type=int,
                        default=2,
                        help='dimensionality of data')
    parser.add_argument('--z_dim',
                        type=int,
                        default=10,
                        help='dimensionality of latent space')
    parser.add_argument("--pi", nargs='*', type=float, default=None,
                        help='mixture weights')
    parser.add_argument('--mean_constant',
                        type=float,
                        default=3.0,
                        help='a constant to set mean')
    parser.add_argument('--n_samples',
                        type=int,
                        default=8000,
                        help='the number of samples')
    parser.add_argument('--mb_size',
                        type=int,
                        default=32,
                        help='the batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help='the learning rate')
    parser.add_argument('--Nsteps',
                        type=int,
                        default=10000,
                        help='the number of iterations')
    parser.add_argument('--dist',
                        type=str,
                        default='Wasserstein',
                        help='Wasserstein/Sliced-Wasserstein/Sinkhorn')
    parser.add_argument('--opt',
                        type=str,
                        default='Adam',
                        help='Adam/AdamW')
    parser.add_argument('--init_weight', default=None,
                        help='initial weight to use (default: None)')
    parser.add_argument('--suffix', type=str, default='',
                        help="suffix for the save folder (default: '')")
    parser.add_argument("--stream_data", action='store_true', help="use streaming data")
    parser.add_argument('--reg',
                        type=float,
                        default=1.0,
                        help='the entropic regularization weight')
    parser.add_argument('--scaling',
                        type=float,
                        default=0.9,
                        help='scaling parameter in geomloss')
    args = parser.parse_args()

    # create output directory
    data_suffix = f'{args.datatype}d{args.dim}'
    if args.datatype == 'GMM':
        data_suffix += f'K{args.K}'
    if args.stream_data:
        data_suffix += 'stream'
    zdim_suffix = ''
    if args.z_dim != 10:
        zdim_suffix = f'_zdim{args.z_dim}'
    bs_suffix = f'_bs{args.mb_size}'
    if args.dist == 'Sliced-Wasserstein':
        bs_suffix += '_sw'
    elif args.dist == 'Sinkhorn':
        bs_suffix += f'_sinkhornReg{args.reg}'
    if args.alpha != 0:
        bs_suffix += f'_a{args.alpha}_L{args.L}'
    if args.opt == 'AdamW':
        bs_suffix += '_AdamW'

    
    imagesdir = os.path.join(args.outdir+data_suffix+zdim_suffix+bs_suffix+args.suffix, f'images_s{args.seed}ds{args.dataseed}')
    chkptdir = os.path.join(args.outdir+data_suffix+zdim_suffix+bs_suffix+args.suffix, f'models_s{args.seed}ds{args.dataseed}')
    resdir = os.path.join(args.outdir+data_suffix+zdim_suffix+bs_suffix+args.suffix, f'results_s{args.seed}ds{args.dataseed}.pickle')
    
    os.makedirs(imagesdir, exist_ok=True)
    os.makedirs(chkptdir, exist_ok=True)
    
    use_cuda = torch.cuda.is_available()
    device = 'cuda:'+str(args.gpu) if use_cuda else 'cpu'
    
    
    seed = args.seed
    dataseed = args.dataseed

    loss_param = None
    if args.dist == 'Sinkhorn':
        loss_param = {'eps': args.reg, 'scaling': args.scaling}
    
    data_param = None
    d = args.dim
    n_samples = args.n_samples
    if args.datatype == 'Gaussian':
        mean=np.zeros(d)
        cov=np.eye(d)
        np.random.seed(dataseed)
        data = np.random.multivariate_normal(mean=mean, cov=cov, size=n_samples)
        data = torch.FloatTensor(data).to(device)
        if args.stream_data:
            data_param = {
                'type': 'Gaussian',
                'mean': mean,
                'cov': cov
            }
    else:
        #### data setting
        K = args.K
        pi = torch.tensor(args.pi)
        pi = pi / pi.sum()
        means = torch.zeros(K,d)
        if K <= d+1:
            for i in range(K-1):
                means[i,i] = args.mean_constant
        else:
            for i in range(d):
                means[i,i] = args.mean_constant
            for i in range(min(d, K-d)):
                means[d+i,i] = -args.mean_constant
        covariances = random_covariances_qr(K, d, seed=dataseed)
        
        data, z = sample_gmm(pi, means, covariances, n_samples=n_samples, seed=dataseed)
        data = data.to(device)
        if args.stream_data:
            data_param = {
                'type': 'GMM',
                'prior': pi,
                'means': means,
                'covs': covariances
            }
    if args.init_weight is not None:
        init = torch.load(args.init_weight)
    else:
        init = None

    # set random seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(seed)
    
    res = train(data, args.mb_size, L=args.L, z_dim=args.z_dim, dist=args.dist, lr=args.lr, Nsteps = args.Nsteps, 
                 device=device, alpha=args.alpha, reg_start=0, useL1=False, init=init, share_theta=False,
               imagesdir=imagesdir, chkptdir=chkptdir, opt=args.opt, data_param = data_param, loss_param = loss_param)
    
    print(f'results.pickle saved in {resdir}.')
    with open(resdir, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    main()