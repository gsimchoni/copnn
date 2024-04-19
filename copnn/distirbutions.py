import numpy as np
from scipy import stats, special


class Distribution:
    def __init__(self, dist_name):
        self.dist_name = dist_name
    
    def __str__(self) -> str:
        return self.dist_name
    
    def sample(self, n, sig2):
        raise NotImplementedError('The sample method is not implemented.')

    def quantile(self, u):
        raise NotImplementedError('The quantile method is not implemented.')
    
    def sum_log_pdf_tf(self, x_tf):
        raise NotImplementedError('The sum_log_pdf_tf method is not implemented.')
    
    def cdf_tf(self, x_tf):
        raise NotImplementedError('The cdf_tf method is not implemented.')

    def cdf_np(self, x_np):
        raise NotImplementedError('The cdf_np method is not implemented.')


class Gaussian(Distribution):
    def __init__(self, dist_name):
        super().__init__(dist_name)
    
    def sample(self, n, sig2):
        return np.random.normal(0, np.sqrt(sig2), n)
        
    def quantile(self, u):
        return stats.norm.ppf(u)
    
    def sum_log_pdf_tf(self, x_tf):
        pass
    
    def cdf_tf(self, x_tf):
        pass

    def cdf_np(self, x_np):
        pass


class Laplace(Distribution):
    def __init__(self, dist_name):
        super().__init__(dist_name)
    
    def sample(self, n, sig2):
        return np.random.laplace(0, np.sqrt(sig2/2), n)
        
    def quantile(self, u):
        return stats.laplace.ppf(u, scale = 1/np.sqrt(2))
    
    def sum_log_pdf_tf(self, x_tf):
        pass
    
    def cdf_tf(self, x_tf):
        pass

    def cdf_np(self, x_np):
        pass


class Exponential(Distribution):
    def __init__(self, dist_name):
        super().__init__(dist_name)
    
    def sample(self, n, sig2):
        return np.random.exponential(np.sqrt(sig2), n) - np.sqrt(sig2)
        
    def quantile(self, u):
        return -(np.log(1 - u) + 1)
    
    def sum_log_pdf_tf(self, x_tf):
        pass
    
    def cdf_tf(self, x_tf):
        pass

    def cdf_np(self, x_np):
        pass


class U2Mixture(Distribution):
    def __init__(self, dist_name):
        super().__init__(dist_name)
    
    def sample(self, n, sig2):
        a = np.sqrt(1.5 * sig2)
        return np.random.uniform(-a, a, n) + np.random.uniform(-a, a, n)
        
    def quantile(self, u):
        return np.sign(u - 0.5) * 2 * np.sqrt(1.5) * (1 - np.sqrt(1 + np.sign(u - 0.5) * (1 - 2*u)))
    
    def sum_log_pdf_tf(self, x_tf):
        pass
    
    def cdf_tf(self, x_tf):
        pass

    def cdf_np(self, x_np):
        pass


class N2Mixture(Distribution):
    def __init__(self, dist_name):
        super().__init__(dist_name)
        self.a = 3
    
    def random_n2(n, sig2, n_sig):
        classes = np.random.binomial(n = 1, p = 0.5, size=n)
        n1 = classes.sum()
        n2 = n - n1
        z = np.zeros(n)
        if n1 > 0:
            z[classes == 1] = np.random.normal(loc = -n_sig * np.sqrt(sig2), scale = np.sqrt(sig2), size = n1)
        if n2 > 0:
            z[classes == 0] = np.random.normal(loc = n_sig * np.sqrt(sig2), scale = np.sqrt(sig2), size = n2)
        return z

    def sample(self, n, sig2):
        single_sig2 = sig2 / (1 + self.a**2)
        return self.random_n2(n, single_sig2, self.a)
        
    def quantile(self, u):
        sig = np.sqrt(1 / (1 + self.a ** 2))
        b = np.zeros(len(u))
        b[u < 0.5] = stats.norm.ppf(2 * u[u < 0.5], loc= -self.a * sig, scale = sig)
        b[u > 0.5] = stats.norm.ppf(2 * u[u > 0.5] - 1, loc= self.a * sig, scale = sig)
        return b
    
    def sum_log_pdf_tf(self, x_tf):
        pass
    
    def cdf_tf(self, x_tf):
        pass

    def cdf_np(self, x_np):
        pass


class Gumbel(Distribution):
    def __init__(self, dist_name):
        super().__init__(dist_name)
        self.c = np.sqrt(6) / np.pi
    
    def sample(self, n, sig2):
        return np.random.gumbel(-np.sqrt(sig2) * self.c * np.euler_gamma, np.sqrt(sig2) * self.c, n)
        
    def quantile(self, u):
        
        return stats.gumbel_r.ppf(u, loc = -self.c * np.euler_gamma, scale = self.c)
    
    def sum_log_pdf_tf(self, x_tf):
        pass
    
    def cdf_tf(self, x_tf):
        pass

    def cdf_np(self, x_np):
        pass


class Logistic(Distribution):
    def __init__(self, dist_name):
        super().__init__(dist_name)
        self.w = np.sqrt(3) / np.pi
    
    def sample(self, n, sig2):
        return np.random.logistic(0, np.sqrt(sig2) * self.w, n)
        
    def quantile(self, u):
        return stats.logistic.ppf(u, scale = self.w)
    
    def sum_log_pdf_tf(self, x_tf):
        pass
    
    def cdf_tf(self, x_tf):
        pass

    def cdf_np(self, x_np):
        pass


class SkewNorm(Distribution):
    def __init__(self, dist_name):
        super().__init__(dist_name)
        self.alpha = 1
        self.xi = -self.alpha * np.sqrt(2 * 1 / (np.pi * (1 + self.alpha**2) - 2 * self.alpha**2))
        self.omega = np.sqrt(np.pi * 1 * (1 + self.alpha**2) / (np.pi * (1 + self.alpha**2) - 2 * self.alpha**2))
    
    def sample(self, n, sig2):
        return stats.skewnorm.rvs(self.alpha, np.sqrt(sig2) * self.xi, np.sqrt(sig2) * self.omega, n)
        
    def quantile(self, u):
        return stats.skewnorm.ppf(u, a = self.alpha, loc = self.xi, scale = self.omega)
    
    def sum_log_pdf_tf(self, x_tf):
        pass
    
    def cdf_tf(self, x_tf):
        pass

    def cdf_np(self, x_np):
        pass


class LogGamma(Distribution):
    def __init__(self, dist_name):
        super().__init__(dist_name)
        self.kappa = 1.42625512
        self.digamma = special.digamma(self.kappa)
        self.trigamma = special.polygamma(1, self.kappa)
    
    def sample(self, n, sig2):
        return stats.loggamma.rvs(self.kappa, -self.digamma, np.sqrt(sig2) / np.sqrt(self.trigamma), n)
        
    def quantile(self, u):
        return stats.loggamma.ppf(u, self.kappa, loc = -self.digamma, scale = 1 / np.sqrt(self.trigamma))
    
    def sum_log_pdf_tf(self, x_tf):
        pass
    
    def cdf_tf(self, x_tf):
        pass

    def cdf_np(self, x_np):
        pass