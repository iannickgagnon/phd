import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def estimate_power_glm_gamma(
    mu1: float, 
    mu2: float, 
    kappa1: float, 
    kappa2: float, 
    alpha: float, 
    n_sim: int,
    n: int = 100
) -> tuple[float, float, float]:
    """
    Estimates statistical power using a Generalized Linear Model (GLM) with a gamma family 
    and log link function. The function simulates data from two gamma distributions, fits 
    a GLM to the data, and calculates the proportion of simulations where the group effect 
    is statistically significant.
    
    Args:
        mu1 (float): Mean of the first gamma distribution.
        mu2 (float): Mean of the second gamma distribution.
        kappa1 (float): Shape parameter (kappa) of the first gamma distribution.
        kappa2 (float): Shape parameter (kappa) of the second gamma distribution.
        alpha (float): Significance level for the hypothesis test.
        n_sim (int): Number of simulations to perform.
        n (int, optional): Sample size for each group in each simulation. Default is 100.
    
    Returns:
        tuple: Estimated power, lower bound of CI, upper bound of CI.
    """

    # To store the number of successful simulations defined as those where the p-value is less than alpha
    successes = 0

    # Create group labels
    group_labels = np.array([0] * n + [1] * n)

    # Perform simulations
    for _ in range(n_sim):
 
        # Generate random samples from gamma distributions
        group1 = stats.gamma.rvs(a=kappa1, scale=mu1 / kappa1, size=n)
        group2 = stats.gamma.rvs(a=kappa2, scale=mu2 / kappa2, size=n)

        # Create a combined dataset
        data = np.concatenate([group1, group2])
        
        # Create a DataFrame for GLM
        df = pd.DataFrame({'value': data, 'group': group_labels})

        # Fit GLM with gamma family and log link
        model = smf.glm(formula='value ~ group', 
                        data=df,
                        family=sm.families.Gamma(sm.families.links.Log())).fit()

        # Wald test on the group coefficient
        p_value = model.pvalues['group']
        if p_value < alpha:
            successes += 1

    # Compute power
    power = successes / n_sim

    # Compute Wilson CI
    ci_low, ci_high = wilson_ci(successes, n_sim, alpha)

    return power, ci_low, ci_high


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Computes the Wilson score confidence interval for a binomial proportion.
    
    Args:
        successes (int): Number of successes.
        n (int): Number of trials.
        alpha (float): Significance level. Default is 0.05 for 95% CI.
    
    Returns:
        tuple: (lower bound, upper bound) of the confidence interval.
    """
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / n
    denominator = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denominator
    return center - margin, center + margin


if __name__ == "__main__":

    """
    This script performs a power analysis simulation using a Generalized Linear Model (GLM) 
    with a gamma family and log link function. The goal is to estimate the statistical power 
    to detect a difference between two groups with different gamma-distributed means.
    
    The simulation involves the following steps:

        1. Generate random samples from two gamma distributions representing two groups.
        2. Fit a GLM to the combined dataset with group as a predictor.
        3. Perform a Wald test on the group coefficient to assess statistical significance.
        4. Repeat the process for a specified number of simulations (n_sim).
        5. Calculate the proportion of simulations where the group effect is statistically significant 
    
    The script also evaluates the power for varying sample sizes and plots a power curve to 
    visualize the relationship between sample size and statistical power.
    
    The power curve is plotted with sample sizes on the x-axis and estimated power on the y-axis, 
    with a horizontal line indicating the target power level (e.g., 0.8).

    The parameters are described in the estimate_power_glm_gamma function docstring.
    """

    # Simulation parameters
    mu_1 = 1.0
    mu_2 = 1.2
    kappa_1 = 2
    kappa_2 = 2
    alpha = 0.05
    n_sim = 400
    sample_sizes = np.arange(100, 301, 50)

    # Change font family to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # Font size to 16
    plt.rcParams['font.size'] = 20

    # Run power simulation
    powers = []
    lower_bounds = []
    upper_bounds = []

    for n in sample_sizes:
        print(f"Estimating power for n = {n}...")
        power, ci_low, ci_high = estimate_power_glm_gamma(mu_1, mu_2, kappa_1, kappa_2, alpha=alpha, n=n, n_sim=n_sim)
        powers.append(power)
        lower_bounds.append(ci_low)
        upper_bounds.append(ci_high)

    # Plot power curve with confidence bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        sample_sizes, powers,
        yerr=[np.array(powers) - np.array(lower_bounds), np.array(upper_bounds) - np.array(powers)],
        fmt='o-', capsize=5, label='GLM Power (95% CI)', color='k'
    )
    plt.axhline(y=0.8, color='red', linestyle='--', label='Target power = 0.8')
    plt.title(' ')
    plt.xlabel('Sample size')
    plt.ylabel('Estimated power')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
