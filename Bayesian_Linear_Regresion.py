import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def run_model():
    np.random.seed(42)
    x = np.linspace(500, 4000, 50)  
    true_intercept = 50000  
    true_slope = 150  
    y = true_intercept + true_slope * x + np.random.normal(0, 20000, size=len(x))  

    with pm.Model() as model:
        beta_0 = pm.Normal("beta_0", mu=0, sigma=1e5)  
        beta_1 = pm.Normal("beta_1", mu=0, sigma=500)  
        sigma = pm.HalfNormal("sigma", sigma=20000)  

        mu = beta_0 + beta_1 * x

        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(2000, return_inferencedata=True)

    az.plot_trace(trace)
    plt.show()

    with model:
        ppc = pm.sample_posterior_predictive(trace)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label="Observed Data")
    plt.plot(x, trace.posterior["beta_0"].values.mean() + trace.posterior["beta_1"].values.mean() * x, 
             color="red", label="Bayesian Fit")
    plt.fill_between(x,
                     np.percentile(ppc.posterior_predictive["y_obs"], 5, axis=(0, 1)),
                     np.percentile(ppc.posterior_predictive["y_obs"], 95, axis=(0, 1)),
                     color="red", alpha=0.3, label="Uncertainty")
    plt.xlabel("House Size (sq ft)")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_model()