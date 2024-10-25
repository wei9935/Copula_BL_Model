import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol
from scipy.stats import norm

def min_volatility_weights(expected_returns, cov_matrix, target):
    n_assets = len(expected_returns)
    expected_returns, cov_matrix = expected_returns.values, cov_matrix.values

    weights = cp.Variable(n_assets)
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))

    # Define constraints
    constraints = [cp.sum(weights) == 1, weights >= 0, cp.sum(expected_returns @ weights) >= target]

    # Formulate and solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Convert optimal weights to Pandas Series
    optimal_weights = pd.Series(weights.value, index=expected_returns.index)

    return optimal_weights

def minCVaR(ret, cov, sim='quasi_MC'):
    if type(ret)==pd.Series:
        ret=np.array(ret)
    confidence_level = 0.95
    np.random.seed(123)
    # y is a simulation of asset returns
    if isinstance(sim, str):
        if sim=='quasi_MC':
            # Generate Sobol sequences
            dim = len(ret)
            sobol = Sobol(d=dim, scramble=True)
            quasi_random_samples = sobol.random_base2(m=int(np.log2(10000)))
            # Transform Sobol sequences to match the multivariate normal distribution
            chol_cov_matrix = np.linalg.cholesky(cov)
            norm_samples = norm.ppf(quasi_random_samples)
            y = norm_samples @ chol_cov_matrix.T + ret
            
        elif sim=='Normal':
            y = np.random.multivariate_normal(ret, cov, 10000)

    else:
        y = sim

    num_assets = ret.shape[0]
    wt = cp.Variable(num_assets) # weights

    alpha = cp.Variable()
    u = cp.Variable(y.shape[0])

    portfolio_returns = y @ wt
    loss = -portfolio_returns

    objective = cp.Minimize(alpha + (1 / (1 - confidence_level)) * cp.sum(u) / y.shape[0])
    constraints = [
        u >= loss - alpha,
        u >= 0,
        cp.sum(wt) == 1,
        wt >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    opt_wt = wt.value
    opt_wt = pd.DataFrame([opt_wt], columns=cov.columns).T
    #optimal_alpha = alpha.value

    return opt_wt

def maxSharpe(ret, cov, target):
    num_assets = ret.shape[0]
    wt = cp.Variable(num_assets)

    port_ret = wt.T*ret
    risk = cp.quad_form(wt, cov)

    objective = cp.Minimize(risk)

    constraints = [
        port_ret >= target,  # Portfolio return constraint
        cp.sum(wt) == 1,  # Full investment constraint
        wt >= 0,  # Non-negative weights (long-only positions)
        wt <= 1  # Weights should not exceed the investment amount
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    opt_wt = wt.value
    opt_wt = pd.DataFrame([opt_wt], columns=cov.columns).T

    return opt_wt

def maxUtility(ret, cov, risk_aversion):
    num_assets = len(ret)
    wt = cp.Variable(num_assets)

    constraints = [
        cp.sum(wt) == 1,
        wt <= 1,
        wt >= 0
        ]

    port_ret = wt.T*ret
    risk = cp.quad_form(wt, cov)
    objective = cp.Maximize(port_ret - 0.5 * risk_aversion * risk)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    opt_wt = wt.value

    return opt_wt

