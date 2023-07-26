import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_fn

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
        self.std = self.df / (self.df - 2)

    def cdf(self, x) -> float:
        return stats.t.cdf(x=x, df=self.df, loc=0.0, scale=1.0)
    
    def log_pdf(self, x) -> float:
        to_sum = [
            np.log(gamma_fn((self.df + 1) / 2)),
            np.log(np.sqrt(self.df * np.pi)),
            np.log(gamma_fn(self.df / 2.0)),
            - 0.5 * (self.df + 1) * np.log(1 + x**2 / self.df)
        ]
        return np.sum(to_sum)
    
    def grad_log_pdf(self, x) -> float:
        return - (self.df + 1) * x / (self.df + x**2)
    
    def ggrad_log_pdf(self, x) -> float:
        return - (self.df + 1) * (self.df - x**2) / (self.df + x**2)**2
    
    def gggrad_log_pdf(self, x) -> float:
        return 2 * (1 + self.df) * x * (3 * self.df - x**2) / (self.df + x**2)**3