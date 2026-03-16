import numpy as np
import math
import torch
import torch.nn as nn
import ot
from geomloss import SamplesLoss

def sample_gmm(pi, means, covariances, n_samples, device="cpu", seed=None):
    """
    Generate samples from a Gaussian Mixture Model using PyTorch.

    Parameters
    ----------
    pi : tensor of shape (K,)
        Mixture weights that sum to 1.
    means : tensor of shape (K, d)
        Means of each Gaussian component.
    covariances : tensor of shape (K, d, d)
        Covariance matrices of each Gaussian component.
    n_samples : int
        Number of data points to generate.
    device : str
        'cpu' or 'cuda'.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : tensor of shape (n_samples, d)
        Sampled data points.
    z : tensor of shape (n_samples,)
        Component labels for each sample.
    """
    if seed is not None:
        torch.manual_seed(seed)

    pi = pi.to(device)
    means = means.to(device)
    covariances = covariances.to(device)

    K, d = means.shape

    # Sample mixture component assignments
    z = torch.multinomial(pi, num_samples=n_samples, replacement=True)

    # Allocate storage
    X = torch.zeros((n_samples, d), device=device)

    for k in range(K):
        idx = (z == k).nonzero(as_tuple=True)[0]
        n_k = idx.shape[0]
        if n_k > 0:
            dist = torch.distributions.MultivariateNormal(
                loc=means[k],
                covariance_matrix=covariances[k]
            )
            X[idx] = dist.sample((n_k,))

    return X, z

def random_covariances_qr(K, d, eig_low=0.1, eig_high=2.0, device=None, dtype=None, seed=None):
    """
    Batch of K SPD matrices via Q diag(λ) Qᵀ with λ ∈ [eig_low, eig_high].
    """
    if seed is not None:
        torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(K, d, d, device=device, dtype=dtype))  # orthonormal
    lam = eig_low + (eig_high - eig_low) * torch.rand(K, d, device=device, dtype=dtype)
    Sigma = Q @ torch.diag_embed(lam) @ Q.transpose(-1, -2)
    return Sigma

def reorder_by_pi(means0, weights0, covs0, pi, descending=True):
    """
    Reorder (means0, weights0, covs0) so that weights0's rank order matches pi's.
    All inputs are NumPy arrays with shapes:
      means0: (K, d), weights0: (K,), covs0: (K, d, d), pi: (K,)
    """
    means0 = np.asarray(means0)
    weights0 = np.asarray(weights0)
    covs0 = np.asarray(covs0)
    pi = np.asarray(pi)

    # Rank orders
    idx_pi = np.argsort(pi, kind="stable")
    idx_w  = np.argsort(weights0, kind="stable")
    if descending:
        idx_pi = idx_pi[::-1]
        idx_w  = idx_w[::-1]

    # Build permutation: for each destination position dst=idx_pi[k],
    # take source component src=idx_w[k]
    perm = np.empty_like(idx_pi)
    perm[idx_pi] = idx_w  # perm[dst] = src

    return means0[perm], weights0[perm], covs0[perm], perm

def gmm_avg_loglik(model: nn.Module, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Average log-likelihood E[log p_model(x)] for a Gaussian mixture with
    means=model.means (K,d), covariances = L^T L from model.Ls (K,d,d),
    and fixed weights model.pi (K,).

    Args:
        model: your Neural_network instance
        x:     (N,d) data
        eps:   jitter added to covariances for numerical stability

    Returns:
        scalar tensor (average log-likelihood over N samples)
    """
    device = x.device
    K, d = model.K, model.dim

    # (K,)
    pi = (model.pi.to(device) / model.pi.to(device).sum()).clamp_min(1e-12)
    log_pi = pi.log()

    # Means and cov factors
    means = model.means.to(device)             # (K,d)
    Ls    = model.Ls.to(device)                # (K,d,d)

    # Build covariances Sigma_k = L_k^T L_k + eps*I and do Cholesky
    I = torch.eye(d, device=device).expand(K, d, d)      # (K,d,d)
    Sigma = torch.matmul(Ls.transpose(-1, -2), Ls) + eps * I  # (K,d,d)
    R = torch.linalg.cholesky(Sigma)                     # (K,d,d), upper-tri

    # Centered data for each component: (N,K,d)
    centered = x[:, None, :] - means[None, :, :]         # (N,K,d)

    # Solve R y = centered^T  -> y has shape (K,d,N)
    # Rearrange to batch by component:
    centered_T = centered.permute(1, 2, 0)               # (K,d,N)
    y = torch.linalg.solve_triangular(R, centered_T, upper=True)  # (K,d,N)

    # Mahalanobis terms: sum over d, then transpose to (N,K)
    maha = (y ** 2).sum(dim=1).permute(1, 0)             # (N,K)

    # log |Sigma_k| from Cholesky: 2 * sum(log diag(R_k))
    log_det = 2.0 * torch.log(torch.diagonal(R, dim1=-2, dim2=-1)).sum(-1)  # (K,)

    const = d * math.log(2.0 * math.pi)
    # log N(x | mu_k, Sigma_k) for all (N,K)
    log_prob_xk = -0.5 * (const + maha + log_det)        # (N,K)

    # Mixture: log sum_k pi_k * N_k = logsumexp_k [log_pi_k + log_prob_xk]
    log_mix = torch.logsumexp(log_prob_xk + log_pi[None, :], dim=1)  # (N,)

    return log_mix.mean()   # scalar

def smoothing(traj, windows=5):
    temp = np.zeros((windows, len(traj)))
    for k in range(windows):
        temp[k,:len(traj)-k] = traj[k:]
    return np.mean(temp, axis=0)[:-windows]

def wass_1d(x, y, p=2) -> torch.Tensor:
    """
    Exact 1D p-Wasserstein distance between two equally-weighted samples.
    x, y: 1D float tensors of the same length.
    """
    x_sorted, _ = torch.sort(x.reshape(-1))
    y_sorted, _ = torch.sort(y.reshape(-1))
    if x_sorted.shape != y_sorted.shape:
        raise ValueError("x and y must have the same number of points.")
    return torch.mean(torch.abs(x_sorted - y_sorted) ** p)

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

def emd(mb_C, mb_size):
    mb_a, mb_b = ot.unif(mb_size), ot.unif(mb_size)
    return ot.emd(mb_a, mb_b, mb_C.detach().cpu().numpy().copy())

def sinkhorn(mb_C, mb_size, reg):
    mb_a, mb_b = ot.unif(mb_size), ot.unif(mb_size)
    return ot.sinkhorn(mb_a, mb_b, mb_C.detach().cpu().numpy().copy(), reg, log=True,
                                numItermax=10000, stopThr=1e-9, method='sinkhorn')

def sinkhorn_divergence(x, y, eps=0.05, p=2, scaling=0.9):
    device = x.device
    blur = eps**(1.0/p)
    loss = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling, debias=True)
    return loss(x, y)

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

def GMM_fitting_score(pi, means, covs, means2, covs2):
    score = 0
    means = means.cpu().numpy()
    covs = covs.cpu().numpy()
    means2 = means2.cpu().numpy()
    covs2 = covs2.cpu().numpy()
    for k in range(len(pi)):
        score += pi[k] * wass_gaussians(means[k], means2[k], covs[k], covs2[k])
    return score.item()

def GMM_fitting_score_from_net(pi, means, covs, net):
    return GMM_fitting_score(pi, means, covs, net.means.detach(), torch.bmm(net.Ls.permute(0,2,1), net.Ls).detach())