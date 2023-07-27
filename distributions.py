import numpy as np
from scipy import stats
from scipy.special import (
    gamma as gamma_fn,
    beta as beta_fn
)

from utils import sigmoid_function

class Dist(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def cdf(self, x) -> float:
        pass

    def log_pdf(self, x) -> float:
        pass

    def grad_log_pdf(self, x) -> float:
        pass

    def ggrad_log_pdf(self, x) -> float:
        pass

    def gggrad_log_pdf(self, x) -> float:
        pass

class NormalDist(Dist):
    def __init__(self, mu, sigma, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = 'Normal distribution'
        self.mu = mu
        self.sigma = sigma

        self.mean = self.mu
        self.std = self.sigma

    def cdf(self, x) -> float:
        return stats.norm.cdf(x, loc=self.mu, scale=self.sigma)
    
    def log_pdf(self, x) -> float:
        to_sum = [
            - np.log(self.sigma),
            - 0.5 * np.log(2 * np.pi),
            - 0.5 * (x - self.mu)**2 / self.sigma**2
        ]
        return np.sum(to_sum)
    
    def grad_log_pdf(self, x) -> float:
        return - (x - self.mu) / self.sigma**2
    
    def ggrad_log_pdf(self, x) -> float:
        return - 1 / self.sigma**2
    
    def gggrad_log_pdf(self, x) -> float:
        return 0.0
    
class StudentTDist(Dist):
    def __init__(self, df, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = 'Student-t distribution'
        self.df = df

        self.mean = 0.0
        self.std = np.sqrt(self.df / (self.df - 2))

    def cdf(self, x) -> float:
        return stats.t.cdf(x=x, df=self.df, loc=0.0, scale=1.0)
    
    def log_pdf(self, x) -> float:
        to_sum = [
            np.log(gamma_fn((self.df + 1) / 2)),
            - np.log(np.sqrt(self.df * np.pi)),
            - np.log(gamma_fn(self.df / 2.0)),
            - 0.5 * (self.df + 1) * np.log(1 + x**2 / self.df)
        ]
        return np.sum(to_sum)
    
    def grad_log_pdf(self, x) -> float:
        return - (self.df + 1) * x / (self.df + x**2)
    
    def ggrad_log_pdf(self, x) -> float:
        return - (self.df + 1) * (self.df - x**2) / (self.df + x**2)**2
    
    def gggrad_log_pdf(self, x) -> float:
        return 2 * (1 + self.df) * x * (3 * self.df - x**2) / (self.df + x**2)**3
    
class GammaDist(Dist):
    def __init__(self, alpha, beta, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

        self.mean = self.alpha / self.beta
        self.std = np.sqrt(self.alpha / self.beta**2)

    def cdf(self, x) -> float:
        return stats.gamma.cdf(x=x, a=self.alpha, loc=0.0, scale=1/self.beta)
    
    def log_pdf(self, x) -> float:
        to_sum = [
            np.log(self.beta**self.alpha / gamma_fn(self.alpha)),
            (self.alpha - 1) * np.log(x),
            - self.beta * x,
        ]
        return np.sum(to_sum)
    
    def grad_log_pdf(self, x) -> float:
        return (self.alpha - 1) / x - self.beta
    
    def ggrad_log_pdf(self, x) -> float:
        return - (self.alpha - 1) * x**(-2)
    
    def gggrad_log_pdf(self, x) -> float:
        return 2 * (self.alpha - 1) * x**(-3)
    
class GeneralisedLogisticDist(Dist):
    def __init__(self, alpha, beta, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        pass

    def log_pdf(self, x) -> float:
        to_sum = [
            self.alpha * np.log(sigmoid_function(x)),
            self.beta * np.log(sigmoid_function(-x)),
            - np.log(beta_fn(self.alpha, self.beta)),
        ]
        return np.sum(to_sum)
    
    # derivative of a * log(1 / (1 + e^(-x))) + b * log(1 / (1 + e^x))
    def grad_log_pdf(self, x) -> float:
        return (self.alpha - self.beta * np.exp(x)) / (np.exp(x) + 1)
    
    # derivative of (a - b e^x)/(1 + e^x)
    def ggrad_log_pdf(self, x) -> float:
        return - np.exp(x) * (self.alpha + self.beta) / (np.exp(x) + 1)**2
    
    # derivative of -((a + b) e^x)/(1 + e^x)^2
    def gggrad_log_pdf(self, x) -> float:
        return np.exp(x) * (np.exp(x) - 1) * (self.alpha + self.beta) / (np.exp(x) + 1)**3