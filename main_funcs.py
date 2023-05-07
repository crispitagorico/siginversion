import numpy as np
from torch.utils.data import Dataset
from fbm import FBM
import iisignature
# esig.stream2sig
import math
from math import gamma
from scipy.integrate import quad
from free_lie_algebra import *
from abc import ABC, abstractmethod


def fbm_generator(sample_num, n, hurst):
    f = FBM(n=n-1, hurst=hurst)
    data = np.zeros((sample_num, n))
    for i in range(sample_num):
        data[i] = f.fbm()
    return data

def rand_sin_generator(sample_num, n, rand_num=20, average=True):
    data = np.zeros((sample_num, n))
    for i in range(sample_num):
        coef = np.random.randn(rand_num) * 5
        # coef = np.random.randint(1, 10, size=rand_num) * np.pi
        data[i] = np.sum(np.sin(np.outer(coef, np.linspace(0, 1, n))), axis=0)
    if average:
        col_f = data[:, 0]*5/6 + data[:, -1]/6
        col_l = data[:, 0]/6 + data[:, -1]*5/6
        data[:, 0] = col_f
        data[:, -1] = col_l
    return data

def sig_AT(path, s_depth, time=None):
    R, n = path.shape
    siglength = iisignature.siglength(2, s_depth)
    if time is None:
        time = np.linspace(0, 1, n)
    sig = np.zeros((R, siglength))
    for i in range(R):
        path_AT = np.vstack((time, path[i])).T
        sig[i] = iisignature.sig(path_AT, s_depth)
    return sig

def fft_aug(path, f_depth):
    if len(path.shape)==1:
        coef = np.fft.rfft(path)[:f_depth]
        return np.concatenate((coef.real, coef.imag))
    else:
        coef = np.fft.rfft(path)[:, :f_depth]
        return np.concatenate((coef.real, coef.imag), axis=1)

def ifft_aug(coef_aug, n, f_depth):
    dimension = coef_aug.shape
    if len(dimension)==1:
        ns = dimension[0]
        coef = coef_aug[:ns//2] + coef_aug[ns//2:] * 1j
        coef_full = np.concatenate((coef, np.zeros(n//2-f_depth+1)))
        return np.fft.irfft(coef_full)
    else:
        m, ns = dimension
        coef = coef_aug[:, :ns//2] + coef_aug[:, ns//2:] * 1j
        coef_full = np.concatenate((coef, np.zeros((m, n//2-f_depth+1))), axis=1)
        return np.fft.irfft(coef_full)

def mse(output, label):
    return np.linalg.norm(output-label)


class sig_Dataset(Dataset):
    def __init__(self, sample_num, n, f_depth, s_depth):
        self.sample_num = sample_num
        self.n = n
        self.f_depth = f_depth
        # np.random.seed(1531)
        # self.path = fbm_generator(sample_num, n, hurst=0.97)
        self.path = rand_sin_generator(sample_num, n, rand_num=15, average=False)
        self.inputs = sig_AT(self.path, s_depth, time=np.linspace(0, 1, n))
        self.labels = fft_aug(self.path, f_depth)
        self.inputs_dim = self.inputs.shape[1]
        self.labels_dim = self.labels.shape[1]

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def reconstruct_path(self, idx=None):
        if idx is None:
            return ifft_aug(self.labels, self.n, self.f_depth)
        else:
            return ifft_aug(self.labels[idx].reshape(1, -1),
                            self.n, self.f_depth).flatten()


class Orthogonal_poly(ABC):
    def __init__(self, x1=-1, x2=1):
        self.x1 = x1
        self.x2 = x2
        self.weight = lambda t: 1

    @abstractmethod
    def P(self, n):
        pass

    @abstractmethod
    def ortho_factor(self, n):
        pass

    @abstractmethod
    def recurrence(self, n):
        pass

    def a(self, x, n):
        P_n = self.P(n)
        snorm_n = self.ortho_factor(n)
        an = quad(lambda t: x(t)*P_n(t)*self.weight(t), self.x1, self.x2)[0]/snorm_n
        return an
    
    def ploy2path(self, x, N, t_grid):
        poly_map = lambda t: sum([self.a(x, n)*self.P(n)(t) for n in range(N+1)])
        return poly_map(t_grid)

    def l(self, n):
        if n==0:
            A0 = self.recurrence(n)
            s_norm0 = self.ortho_factor(n)
            return A0 / s_norm0 * word2Elt('21') 
        elif n==1:
            A1, B1 = self.recurrence(n)
            s_norm1 = self.ortho_factor(n)
            return (A1*self.x1+B1)/(s_norm1)*word2Elt('21') + A1/s_norm1*(word2Elt('211')+word2Elt('121'))
        else:
            An, Bn, Cn = self.recurrence(n)
            snorm_n = self.ortho_factor(n)
            snorm_n1 = self.ortho_factor(n-1)
            snorm_n2 = self.ortho_factor(n-2)
            l_prev = self.l(n-1)
            l_prev_prev = self.l(n-2)
            l_n = An*snorm_n1/snorm_n*rightHalfShuffleProduct(word2Elt('1'),l_prev)\
                  + (An*self.x1+Bn)*snorm_n1/snorm_n*l_prev\
                  + Cn*snorm_n2/snorm_n*l_prev_prev
            return l_n

    def a_sig(self, sig, n):
        return dotprod(self.l(n), sig)

    def sig2path(self, sig, N, t_grid):
        sig_map =  lambda t: sum([self.a_sig(sig, n)*self.P(n)(t) for n in range(N+1)])
        return sig_map(t_grid) / self.weight(t_grid)


class Jacobi(Orthogonal_poly):
    def __init__(self, alpha, beta):
        super().__init__(-1, 1)
        self.alpha = alpha
        self.beta = beta
        self.weight = lambda t: (1-t)**alpha*(1+t)**beta

    def P(self, n):
        if n==0:
            return lambda t: 0*t + 1
        else:
            a = self.alpha
            b = self.beta
            factor = gamma(a+n+1) / math.factorial(n) / gamma(a+b+n+1)
            return lambda t: factor * sum([math.comb(n, m)*gamma(a+b+n+m+1)/gamma(a+m+1)*((t-1)/2)**m for m in range(n+1)])

    def ortho_factor(self, n):
        a = self.alpha
        b = self.beta
        return 2**(a+b+1) * gamma(n+a+1) * gamma(n+b+1) / (2*n+a+b+1) / gamma(n+a+b+1) / math.factorial(n)

    def recurrence(self, n):
        a = self.alpha
        b = self.beta
        if n == 0:
            return 1
        elif n==1:
            A1 = (a+b+2) / 2
            B1 = (a+1) - A1
            return A1, B1
        else:
            x = n + a
            y = n + b
            z = x + y
            denominator = 2 * n * (z-n) * (z-2)
            An = z * (z-1) * (z-2) / denominator
            Bn = (z-1) * (x-y) * (z-2*n) / denominator
            Cn = - 2 * (x-1) * (y-1) * z / denominator
            # verify recurrence relation
            # P = self.P
            # grid = np.random.uniform(-1, 1, 30)
            # assert(np.linalg.norm(P(n)(grid)-(An*grid+Bn)*P(n-1)(grid)-Cn*P(n-2)(grid))<1.0e-6)
            return An , Bn, Cn


class Legendre(Jacobi):
    def __init__(self):
        super().__init__(0, 0)
        self.weight = lambda t: 1

    def ortho_factor(self, n):
        return 2 / (2*n+1)


class Chebyshev(Jacobi):
    def __init__(self):
        super().__init__(-0.5, -0.5)
    
    def ortho_factor(self, n):
        if n==0:
            return math.pi
        else:
            return math.pi/2


class Hermite(Orthogonal_poly):
    def __init__(self, t0, eps, start_poiint=None):
        if start_poiint is None:
            start_poiint = -np.inf
        super().__init__(start_poiint, np.inf)
        self.t0 = t0
        self.eps = eps
        self.weight = lambda t: np.exp(-(t-t0)**2/2/eps**2)

    def P(self, n):
        return lambda t: math.factorial(n)*sum([(-1)**m*((t-self.t0)/self.eps)**(n-2*m)/2**m/math.factorial(m)/math.factorial(n-2*m) for m in range(n//2+1)])
    
    def ortho_factor(self, n):
        return math.sqrt(2*math.pi)*self.eps*math.factorial(n)
    
    def recurrence(self, n):
        if n == 0:
            return 1
        elif n == 1:
            return 1/self.eps, -self.t0/self.eps
        An, Bn, Cn = 1/self.eps, -self.t0/self.eps, -n+1
        # check recurrence relation of Hermite polynomial
        # P = self.P
        # grid = np.random.uniform(-1, 1, 30)
        # assert(np.linalg.norm(P(n)(grid)-(An*grid+Bn)*P(n-1)(grid)-Cn*P(n-2)(grid))<1.0e-6)
        return An, Bn, Cn


class Fourier():
    def __init__(self, t_grid):
        self.t_grid = list(t_grid)

    def ploy2path(self, x, N, t0):
        idx = self.t_grid.index(t0)
        coeff = fft_aug(np.array([x(t) for t in self.t_grid]), N)
        return ifft_aug(coeff, len(self.t_grid), N)[idx]