import numpy as np

def linear_shrinkage_cov(cov, alpha):
    """
    Cleans a covariance matrix using linear shrinkage.

    Parameters:
    - cov: (numpy.ndarray) Input covariance matrix (N x N)
    - alpha: (float) Shrinking intensity
    
    Returns:
    - (numpy.ndarray) Cleaned covariance matrix using shrinkage
    """
    return (1-alpha)*cov + (alpha)*np.diag(np.diag(cov))

def clipping_cov(cov, T, N):
    """
    Cleans a covariance matrix using RMT, more specifically using eigenvalue clipping.
    
    Parameters:
    - cov: (numpy.ndarray) Input covariance matrix (N x N)
    - T: (int) Number of observations
    - N: (int) Number of variables (dimensionality)
    
    Returns:
    - Cov_clip: (numpy.ndarray) Cleaned covariance matrix using clipping
    """
    # Calculate the quality factor
    q = N / T
    # Lambda max for random correlation matrix
    lmax = (1 + np.sqrt(q)) ** 2
    
    # We convert our covariance matrix to correlation matrix
    std_array = np.diag(cov)
    std_diag_inv = np.diag(1 / np.sqrt(std_array))
    Corr = std_diag_inv @ cov @ std_diag_inv

    # We clean the correlation matrix
    eig = np.linalg.eigh(Corr)
    u_hat = eig.eigenvectors
    l_hat = eig.eigenvalues
    l_hat[l_hat <= lmax] = np.mean(l_hat[l_hat < lmax])
    Corr_clip = u_hat @ np.diag(l_hat) @ u_hat.T

    # We ensure the diagonal of the correlation matrix is only ones
    diag_corr = np.diag(1 / np.sqrt(np.diag(Corr_clip)))
    Corr_clip = diag_corr @ Corr_clip @ diag_corr

    # We convert back the correlation matrix to covariance matrix
    std_diag = np.diag(std_array)  
    Cov_clip = std_diag @ Corr_clip @ std_diag
    return Cov_clip

def calculate_portfolio_return(rolling_size, keeping_days, returns_df, T, N, cleaning_recipe = 0):
    """
    Calculate portfolio returns using mean-variance optimization.

    Parameters:
    - rolling_size: (int) Window size for rolling covariance calculation.
    - keeping_days: (int) Number of out sample days we keep the portfolio weight
    - returns_df: (pd.DataFrame) DataFrame of asset returns (T x N).
    - T: (int) Total number of time periods.
    - N: (int) Number of assets.
    - cleaning_recipe: (int) Method used to clean the covariance matrix.

    Returns:
    - portfolio_returns: (list) Portfolio returns for each period.
    """
    portfolio_return = np.array([])
    ones_N = np.ones((N))
    for i in range(rolling_size - 1, T - 1 - keeping_days, keeping_days):
        
        # covariance matrix
        cov_i = returns_df.loc[i - rolling_size + 1:i].cov()
        # mean_variance weight
        if cleaning_recipe == 1:
            cov_i_inv = np.linalg.inv(clipping_cov(cov_i, rolling_size, N))
        elif cleaning_recipe == 2:
            cov_i_inv = np.linalg.inv(linear_shrinkage_cov(cov_i, 0.5))
        elif cleaning_recipe == 3:
            cov_i_inv = np.linalg.inv(linear_shrinkage_cov(cov_i, 1))
        else:
            cov_i_inv = np.linalg.inv(cov_i)
        
        w_i = cov_i_inv @ ones_N / (ones_N @ cov_i_inv @ ones_N)
        
        # We remove the negative weight then rescale
        w_i[w_i < 0] = 0
        w_i = w_i / np.sum(w_i) 

        # return of stocks on day i + 1
        ret_i_1 = np.array(returns_df.loc[i + 1:i + keeping_days])

        # return of our portfolio on day i + 1
        portfolio_return = np.concatenate((portfolio_return, ret_i_1 @ w_i))
    return portfolio_return
