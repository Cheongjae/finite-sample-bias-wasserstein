import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

from utils_GMMfit import sample_gmm, random_covariances_qr, reorder_by_pi, wass_1d, sliced_wasserstein, \
emd, sinkhorn_divergence, squared_distances, wass_gaussians, GMM_fitting_score, GMM_fitting_score_from_net, gmm_avg_loglik

class Neural_network(nn.Module):
    """
    Simple GMM sampler with fixed pi.
    forward(x) expects x ~ N(0, I) of shape (N, dim) and returns samples.
    """
    def __init__(self, K, dim, pi, init=None):
        super().__init__()
        self.K, self.dim = K, dim
        self.pi = torch.as_tensor(pi, dtype=torch.float)          # fixed, not a Parameter
        if init is None:
            self.means = nn.Parameter(torch.randn(K, dim) * 0.1)
            # Start Ls as identity so each component is non-degenerate
            self.Ls = nn.Parameter(torch.eye(dim).repeat(K, 1, 1))    # (K, dim, dim)
        else:
            self.means = nn.Parameter(torch.FloatTensor(init['means']))
            # Start Ls as identity so each component is non-degenerate
            self.Ls = nn.Parameter(torch.FloatTensor(init['Ls']))    # (K, dim, dim)

    def forward(self, x, return_c=False):
        # x: (N, dim), base Gaussian noise (typically standard normal)
        N, d = x.shape
        p = (self.pi / self.pi.sum()).to(x.device)                # ensure proper probs & device
        z = torch.multinomial(p, num_samples=N, replacement=True) # (N,)

        Lz  = self.Ls[z]          # (N, dim, dim)
        muz = self.means[z]       # (N, dim)

        # y = x @ L + mu  (batched)
        y = torch.bmm(x.unsqueeze(1), Lz).squeeze(1) + muz
        if return_c:
            return y, z
        return y

def get_entropy(pi):
    return (pi * pi.clamp_min(1e-12).log()).sum().item()

def train(means, covs, prior, mb_size, reg, scaling, lr, Nsteps, device, data = None, nt = 10000, alpha=None, dist='Wasserstein', share_theta=False, init=None, seed=None):
    # Parameters for the gradient descent
    if seed is not None:
        torch.manual_seed(seed)
    save_its = [int(Nsteps/8), int(Nsteps/4), int(Nsteps/2), Nsteps-1] #plot results for wanted iteration
    
    loss_error = []
    score = []
    fx_set = []

    individual_loss= dict()
    Cxy_set = []
    Cxx_set = []
    Cyy_set = []
    Cxx_indep_set = []
    Cxx2_set = []
    Sxy_set = []
    Sxx_indep_set = []
    ent_xy_set = []
    ent_xx_set = []
    ent_yy_set = []
    ent_xx_indep_set = []
    ent_xx2_set = []
    errors_set = []

    K = len(prior)
    dim = means.shape[1]
    
    if data is not None:
        # Make sure that we won't modify the reference samples
        data = data.clone()
        data = data.to(device)
        nt = data.size()[0]

    net = Neural_network(K, dim, prior, init)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    score.append(GMM_fitting_score_from_net(prior, means, covs, net))
    
    for i in range(Nsteps): # Euler scheme ===============
        optimizer.zero_grad()
        temp_error = []

        mb_x = torch.randn(mb_size, dim).type(torch.FloatTensor).to(device) #Batch source
        if data is None:
            mb_y, _ = sample_gmm(prior, means, covs, n_samples=mb_size)
            mb_y = torch.FloatTensor(mb_y).to(device)
        else:
            mb_t = np.random.choice(nt, size=mb_size, replace=False) #Batch target
            mb_y = data[mb_t]

        if dim == 1:
            f_x = net(mb_x) # G(z)
            Loss = wass_1d(f_x, mb_y)
            if alpha is not None:
                mb_x2 = torch.randn(mb_size, dim).type(torch.FloatTensor).to(device) #Batch source
                f_x2 = net(mb_x2) # G(z)
                Loss -= alpha*wass_1d(f_x, f_x2)
        elif dist == 'Wasserstein' and reg == 0.0:
            f_x = net(mb_x) # G(z)
            # C = torch.sqrt(squared_distances(f_x, mb_y))
            C = squared_distances(f_x, mb_y)
            pi = torch.as_tensor(emd(C, mb_size)).to(device)
            Cxy = torch.sum(pi * C.double())
            Loss = Cxy
            Cxy_set.append(Cxy.item())
            ent_xy_set.append(get_entropy(pi))
            if alpha is not None:
                mb_x2 = torch.randn(mb_size, dim).type(torch.FloatTensor).to(device) #Batch source
                f_x2 = net(mb_x2) # G(z)
                C2 = squared_distances(f_x, f_x2)
                pi2 = torch.as_tensor(emd(C2, mb_size)).to(device)
                Cxx_indep = torch.sum(pi2 * C2.double())
                Loss -= alpha*Cxx_indep
                Cxx_indep_set.append(Cxx_indep.item())
                ent_xx_indep_set.append(get_entropy(pi2))
        elif dist == 'Wasserstein' and reg > 0.0:
            f_x = net(mb_x) # G(z)
            Sxy = sinkhorn_divergence(f_x, mb_y, eps=reg, p=2, scaling=scaling)
            Loss = Sxy
            Sxy_set.append(Sxy.item())
            if alpha is not None:
                mb_x2 = torch.randn(mb_size, dim).type(torch.FloatTensor).to(device) #Batch source
                f_x2 = net(mb_x2) # G(z)
                Sxx_indep = sinkhorn_divergence(f_x, f_x2, eps=reg, p=2, scaling=scaling)
                Loss -= alpha*Sxx_indep
                Sxx_indep_set.append(Sxx_indep.item())
        elif dist == 'MLE':
            Loss = -gmm_avg_loglik(net, mb_y)
        elif dist == 'Sliced-Wasserstein':
            f_x = net(mb_x) # G(z)
            Loss, theta = sliced_wasserstein(f_x, mb_y)
            if not share_theta:
                theta = None
            if alpha is not None:
                mb_x2 = torch.randn(mb_size, dim).type(torch.FloatTensor).to(device) #Batch source
                f_x2 = net(mb_x2) # G(z)
                temp, _ = sliced_wasserstein(f_x, f_x2, theta=theta)
                Loss -= alpha*temp
        else:
            raise NotImplementedError
        
        Loss.backward() # Compute gradient
        optimizer.step()

        with torch.no_grad():
            loss_error.append(Loss.item())
            score.append(GMM_fitting_score_from_net(prior, means, covs, net))

        if i in save_its :
            mb_x = torch.randn(nt, dim).type(torch.FloatTensor).to(device)
            f_x = net(mb_x).detach().cpu().numpy()
            fx_set.append(f_x)

    individual_loss['Cxy'] = Cxy_set
    individual_loss['Cxx'] = Cxx_set
    individual_loss['Cyy'] = Cyy_set
    individual_loss['Cxx_indep'] = Cxx_indep_set
    individual_loss['Cxx2'] = Cxx2_set
    individual_loss['Sxy'] = Sxy_set
    individual_loss['Sxx_indep'] = Sxx_indep_set
    individual_loss['ent_xy'] = ent_xy_set
    individual_loss['ent_xx'] = ent_xx_set
    individual_loss['ent_yy'] = ent_yy_set
    individual_loss['ent_xx_indep'] = ent_xx_indep_set
    individual_loss['ent_xx2'] = ent_xx2_set
    individual_loss['errors'] = np.array(errors_set)
    
    return loss_error, fx_set, net, score, individual_loss


def run_experiment(args):
    #### values from args
    device = args.dev
    dataseed = args.dataseed
    K = args.K
    d = args.dim
    pi = torch.tensor(args.pi)
    mean_constant = args.mean_constant
    n_samples = args.n_samples
    seed = args.seed
    mb_size = args.mb_size
    lr = args.lr
    Nsteps = args.Nsteps
    dist = args.dist
    reg = args.reg
    debias = args.debias
    fix_data = args.fix_data
    use_init = args.use_init
    scaling = args.scaling

    #### data setting
    pi = pi / pi.sum()
    means = torch.zeros(K,d)
    if K <= d+1:
        for i in range(K-1):
            means[i,i] = args.mean_constant
    else:
        for i in range(d):
            means[i,i] = mean_constant
        for i in range(min(d, K-d)):
            means[d+i,i] = -mean_constant
    covariances = random_covariances_qr(K, d, seed=dataseed)
    
    X, z = sample_gmm(pi, means, covariances, n_samples=n_samples, seed=dataseed)
    X_np = X.cpu().numpy()
    z_np = z.cpu().numpy()
    
    #### k-means initial guess
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
    labels = km.labels_
    means0 = km.cluster_centers_
    weights0 = np.bincount(labels, minlength=K) / len(X)
    
    # Optional: covariances per cluster (with small jitter)
    covs0 = np.stack([
        ((X[labels==k] - means0[k]).T @ (X[labels==k] - means0[k])) /
        max((labels==k).sum(), 1) + 1e-6 * np.eye(d)
        for k in range(K)
    ], axis=0)

    m2, w2, S2, perm = reorder_by_pi(means0, weights0, covs0, pi.cpu().numpy(), descending=True)


    #### GMM for comparison
    # Fit GMM with K=2
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",   # 'full' | 'tied' | 'diag' | 'spherical'
        init_params="kmeans",
        n_init=5,                 # multiple restarts for robustness
        reg_covar=1e-6,           # jitter for stability
        random_state=42
    ).fit(X_np)
    
    # Cluster labels and responsibilities
    labels = gmm.predict(X_np)          # shape (N,)
    resp   = gmm.predict_proba(X_np)    # shape (N, K)
    m3, w3, S3, perm = reorder_by_pi(gmm.means_, gmm.weights_, gmm.covariances_, pi.cpu().numpy(), descending=True)

    #### train GMM parameters with fixed prior
    data = None
    alpha = None
    init = None
    if fix_data:
        data = X
    if debias:
        alpha = 0.5
    if use_init:
        # use the results from k-means
        init = dict()
        init['weight'] = w2
        init['means'] = m2
        init['Ls'] = np.linalg.cholesky(S2).transpose(0,2,1)
    start = time.time()
    loss_error, fx_set, net, score, individual_loss = train(means, covariances, pi, mb_size, reg, scaling, lr, Nsteps, device, data=data, nt=n_samples, alpha=alpha, dist=dist, init=init, seed=seed)
    elapsed_time = time.time() - start
    print(f'elapsed time: {elapsed_time} seconds.')

    #### gather results
    results = dict()
    results['elapsed_time'] = elapsed_time
    results['loss'] = loss_error
    results['score'] = score
    results['net'] = net
    results['fx_set'] = fx_set
    results['individual_loss'] = individual_loss
    results['data'] = X
    results['score_kmeans'] = GMM_fitting_score(pi, means, covariances, torch.tensor(m2), torch.tensor(S2))
    results['score_GMM'] = GMM_fitting_score(pi, means, covariances, torch.tensor(m3), torch.tensor(S3))
    results['args'] = args
    
    return results