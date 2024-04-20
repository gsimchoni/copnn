import numpy as np
from scipy import stats, special
import tensorflow as tf
import tensorflow.keras.backend as K

class Distribution:
    def __init__(self, dist_par):
        self.dist_par = dist_par
    
    def __str__(self):
        return self.dist_par
    
    def sample(self, n, sig2):
        raise NotImplementedError('The sample method is not implemented.')

    def quantile(self, u):
        raise NotImplementedError('The quantile method is not implemented.')
    
    def cdf(self, x):
        raise NotImplementedError('The cdf method is not implemented.')
    
    def sum_log_pdf_batch(self, x, sig2, n_batch):
        raise NotImplementedError('The sum_log_pdf_batch method is not implemented.')
    
    def cdf_batch(self, x, sig2):
        raise NotImplementedError('The cdf_batch method is not implemented.')


class Gaussian(Distribution):
    def __init__(self):
        super().__init__('gaussian')
    
    def sample(self, n, sig2):
        return np.random.normal(0, np.sqrt(sig2), n)
        
    def quantile(self, u):
        return stats.norm.ppf(u)
    
    def cdf(self, x):
        return stats.norm.cdf(x)

    def sum_log_pdf_batch(self, x, sig2, n_batch):
        z = x / K.sqrt(sig2)
        return K.dot(K.transpose(z), z) + n_batch * np.log(2 * np.pi) + n_batch * tf.math.log(sig2)
    
    def cdf_batch(self, x, sig2):
        z = x / K.sqrt(sig2)
        return 0.5 * (tf.math.erf(z / np.sqrt(2)) + 1)


class Laplace(Distribution):
    def __init__(self):
        super().__init__('laplace')
    
    def sample(self, n, sig2):
        return np.random.laplace(0, np.sqrt(sig2/2), n)
        
    def quantile(self, u):
        return stats.laplace.ppf(u, scale = 1/np.sqrt(2))
    
    def cdf(self, x):
        return stats.laplace.cdf(x, scale = 1/np.sqrt(2))

    def sum_log_pdf_batch(self, x, sig2, n_batch):
        b = K.sqrt(sig2/2)
        z = x / b
        return 2 * tf.reduce_sum(tf.abs(z)) + 2 * n_batch * tf.math.log(2*b)
    
    def cdf_batch(self, x, sig2):
        b = K.sqrt(sig2/2)
        z = x / b
        return 0.5 + 0.5 * tf.sign(z) * (1 - tf.exp(-tf.abs(z)))


class Exponential(Distribution):
    def __init__(self):
        super().__init__('exponential')
    
    def sample(self, n, sig2):
        return np.random.exponential(np.sqrt(sig2), n) - np.sqrt(sig2)
        
    def quantile(self, u):
        return -(np.log(1 - u) + 1)
    
    def cdf(self, x):
        return 1 - np.exp(-x + x.min())

    def sum_log_pdf_batch(self, x, sig2, n_batch):
        sq_sig2e_sig2b = tf.reduce_min(x)
        z = x / K.sqrt(sig2)
        return 2 * tf.reduce_sum(z) - 2 * n_batch * sq_sig2e_sig2b / K.sqrt(sig2) + n_batch * tf.math.log(sig2)
    
    def cdf_batch(self, x, sig2):
        sq_sig2e_sig2b = tf.reduce_min(x)
        z = x / K.sqrt(sig2)
        return 1 - tf.exp(-z + sq_sig2e_sig2b / K.sqrt(sig2))


class U2Mixture(Distribution):
    def __init__(self):
        super().__init__('u2mixture')
    
    def sample(self, n, sig2):
        a = np.sqrt(1.5 * sig2)
        return np.random.uniform(-a, a, n) + np.random.uniform(-a, a, n)
        
    def quantile(self, u):
        return np.sign(u - 0.5) * 2 * np.sqrt(1.5) * (1 - np.sqrt(1 + np.sign(u - 0.5) * (1 - 2*u)))
    
    def cdf(self, x):
        a = np.sqrt(1.5)
        z = np.clip(x, -2*a + 1e-5, 2*a - 1e-5)
        return 0.5 + z / (2 * a) - np.sign(z) * ((z ** 2)/ (8 * (a ** 2)))

    def sum_log_pdf_batch(self, x, sig2, n_batch):
        a = K.sqrt(1.5 * sig2)
        z = tf.clip_by_value(x, -2*a + 1e-5, 2*a - 1e-5)
        return -2 * tf.reduce_sum(tf.math.log((2 * a - tf.abs(z))/(4 * (a ** 2))))
    
    def cdf_batch(self, x, sig2):
        a = K.sqrt(1.5 * sig2)
        z = tf.clip_by_value(x, -2*a + 1e-5, 2*a - 1e-5)
        return 0.5 + z / (2 * a) - tf.sign(z) * ((z ** 2)/ (8 * (a ** 2)))


class N2Mixture(Distribution):
    def __init__(self):
        super().__init__('n2mixture')
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
        b = np.zeros(u.shape)
        b[u < 0.5] = stats.norm.ppf(2 * u[u < 0.5], loc= -self.a * sig, scale = sig)
        b[u > 0.5] = stats.norm.ppf(2 * u[u > 0.5] - 1, loc= self.a * sig, scale = sig)
        return b
    
    def cdf(self, x):
        sig = 1
        z = x * np.sqrt(1 + self.a**2) / sig
        z1 = z - self.a
        z2 = z + self.a
        return 0.5 * (special.erf(z1 / np.sqrt(2)) + 1)/2 + 0.5 * (special.erf(z2 / np.sqrt(2)) + 1)/2

    def sum_log_pdf_batch(self, x, sig2, n_batch):
        sig = K.sqrt(sig2)
        z = x * np.sqrt(1 + self.a**2)/sig - tf.sign(x) * self.a
        return K.dot(K.transpose(z), z) - 2 * n_batch * np.log(2) - n_batch * np.log(1 + self.a**2) + n_batch * tf.math.log(8*np.pi*sig2)
    
    def cdf_batch(self, x, sig2):
        sig = K.sqrt(sig2)
        z = x * np.sqrt(1 + self.a**2) / sig
        z1 = z - self.a
        z2 = z + self.a
        return 0.5 * (tf.math.erf(z1 / np.sqrt(2)) + 1)/2 + 0.5 * (tf.math.erf(z2 / np.sqrt(2)) + 1)/2


class Gumbel(Distribution):
    def __init__(self):
        super().__init__('gumbel')
        self.c = np.sqrt(6) / np.pi
        self.c2 = 6 / (np.pi**2)
    
    def sample(self, n, sig2):
        return np.random.gumbel(-np.sqrt(sig2) * self.c * np.euler_gamma, np.sqrt(sig2) * self.c, n)
        
    def quantile(self, u):
        return stats.gumbel_r.ppf(u, loc = -self.c * np.euler_gamma, scale = self.c)
    
    def cdf(self, x):
        return stats.gumbel_r.cdf(x, loc = -self.c * np.euler_gamma, scale = self.c)

    def sum_log_pdf_batch(self, x, sig2, n_batch):
        z = x / (self.c * K.sqrt(sig2)) + np.euler_gamma
        return 2 * tf.reduce_sum(z) + 2 * tf.reduce_sum(tf.exp(-z)) + n_batch * tf.math.log(sig2) + n_batch * tf.math.log(self.c2)
    
    def cdf_batch(self, x, sig2):
        z = x / (self.c * K.sqrt(sig2)) + np.euler_gamma
        return tf.exp(-tf.exp(-z))


class Logistic(Distribution):
    def __init__(self):
        super().__init__('logistic')
        self.d = np.sqrt(3) / np.pi
        self.d2 = 3 / (np.pi**2)
    
    def sample(self, n, sig2):
        return np.random.logistic(0, np.sqrt(sig2) * self.d, n)
        
    def quantile(self, u):
        return stats.logistic.ppf(u, scale = self.d)
    
    def cdf(self, x):
        return stats.logistic.cdf(x, scale = self.d)

    def sum_log_pdf_batch(self, x, sig2, n_batch):
        z = x / (self.d * K.sqrt(sig2))
        return 2 * tf.reduce_sum(z) + 4 * tf.reduce_sum(tf.math.log(1 + tf.exp(-z))) + n_batch * tf.math.log(sig2) + n_batch * tf.math.log(self.d2)
    
    def cdf_batch(self, x, sig2):
        z = x / (self.d * K.sqrt(sig2))
        return 1 / (1 + tf.exp(-z))


class SkewNorm(Distribution):
    def __init__(self):
        super().__init__('skewnorm')
        self.alpha = 1
        self.xi = -self.alpha * np.sqrt(2 * 1 / (np.pi * (1 + self.alpha**2) - 2 * self.alpha**2))
        self.omega = np.sqrt(np.pi * 1 * (1 + self.alpha**2) / (np.pi * (1 + self.alpha**2) - 2 * self.alpha**2))
    
    def owens_t(self, h):
        return tf.numpy_function(special.owens_t, [h, self.alpha], tf.float32)
    
    def sample(self, n, sig2):
        return stats.skewnorm.rvs(self.alpha, np.sqrt(sig2) * self.xi, np.sqrt(sig2) * self.omega, n)
        
    def quantile(self, u):
        return stats.skewnorm.ppf(u, a = self.alpha, loc = self.xi, scale = self.omega)
    
    def cdf(self, x):
        return stats.skewnorm.cdf(x, a = self.alpha, loc = self.xi, scale = self.omega)
    
    def sum_log_pdf_batch(self, x, sig2, n_batch):
        sig = K.sqrt(sig2)
        z = (x - self.xi * sig) / (sig * self.omega)
        phi_alpha = tf.clip_by_value((tf.math.erf(self.alpha * z / np.sqrt(2)) + 1)/2, 1e-5, np.infty)
        log_phi_minus_times_2 = K.dot(K.transpose(z), z) + n_batch * np.log(2 * np.pi) + n_batch * tf.math.log(sig2) - 2 * n_batch * np.log(2)
        return log_phi_minus_times_2 - 2 * tf.reduce_sum(tf.math.log(phi_alpha))
    
    def cdf_batch(self, x, sig2):
        sig = K.sqrt(sig2)
        z = (x - self.xi * sig) / (sig * self.omega)
        phi = (tf.math.erf(z / np.sqrt(2)) + 1)/2
        # for alpha=1: phi - 2 * T_owen is really phi^2
        return phi**2
        # T_owen = 0.5 * phi * (1 - phi)
        # T_owen = self.owens_t(y)
        # return phi - 2 * T_owen


class LogGamma(Distribution):
    def __init__(self):
        super().__init__('loggamma')
        self.kappa = 1.42625512
        self.digamma = special.digamma(self.kappa)
        self.trigamma = special.polygamma(1, self.kappa)
        self.log_trigamma = np.log(self.trigamma)
        self.log_Gamma_kappa = np.log(special.gamma(self.kappa))
    
    def sample(self, n, sig2):
        return stats.loggamma.rvs(self.kappa, -self.digamma, np.sqrt(sig2) / np.sqrt(self.trigamma), n)
        
    def quantile(self, u):
        return stats.loggamma.ppf(u, self.kappa, loc = -self.digamma, scale = 1 / np.sqrt(self.trigamma))
    
    def cdf(self, x):
        return stats.loggamma.cdf(x, self.kappa, loc = -self.digamma, scale = 1 / np.sqrt(self.trigamma))

    def sum_log_pdf_batch(self, x, sig2, n_batch):
        sig = K.sqrt(sig2 / self.trigamma)
        z = (x + self.digamma) / sig
        return -2 * self.kappa * tf.reduce_sum(z) + 2 * tf.reduce_sum(tf.math.exp(z)) - n_batch * self.log_trigamma + n_batch * tf.math.log(sig2) + 2 * n_batch * self.log_Gamma_kappa
    
    def cdf_batch(self, x, sig2):
        sig = K.sqrt(sig2 / self.trigamma)
        z = (x + self.digamma) / sig
        return tf.math.igamma(self.kappa, tf.math.exp(z))


def get_distribution(marginal):
    if marginal == 'gaussian':
        dist = Gaussian()
    elif marginal == 'laplace':
        dist = Laplace()
    elif marginal == 'exponential':
        dist = Exponential()
    elif marginal == 'u2mixture':
        dist = U2Mixture()
    elif marginal == 'n2mixture':
        dist = N2Mixture()
    elif marginal == 'gumbel':
        dist = Gumbel()
    elif marginal == 'logistic':
        dist = Logistic()
    elif marginal == 'skewnorm':
        dist = SkewNorm()
    elif marginal == 'loggamma':
        dist = LogGamma()
    else:
        raise NotImplementedError(f'{marginal} distribution not implemented.')
    return dist
