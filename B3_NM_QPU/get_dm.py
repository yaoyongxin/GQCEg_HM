import numpy as np
from scipy.optimize import minimize



def refine_1PDM_maxent(gamma_noisy, N, tol=1e-6, max_iter=1000):
    """
    Refine a noisy one-particle density matrix (1PDM) using the Maximum Entropy (MaxEnt) method.

    Parameters:
    - gamma_noisy: np.ndarray
        Noisy 1PDM (Hermitian but possibly unphysical due to noise)
    - N: int
        Expected particle number (trace constraint)
    - tol: float
        Convergence tolerance for optimization
    - max_iter: int
        Maximum number of iterations

    Returns:
    - gamma_refined: np.ndarray
        Refined 1PDM satisfying physical constraints
    """

    # Step 1: Ensure Hermiticity
    gamma_noisy = (gamma_noisy + gamma_noisy.T.conj()) / 2

    # Step 2: Eigenvalue Decomposition
    eigvals, eigvecs = np.linalg.eigh(gamma_noisy)
    eigvals = eigvals.max() - eigvals


    # Step 3: Define the entropy-based function to minimize
    def entropy_loss(beta):
        """Compute the entropy loss for a given beta (Lagrange multiplier)."""
        exp_eigvals = np.exp(-beta * eigvals)
        res = np.abs(np.sum(exp_eigvals) - N)
        # print("err:", res, np.sum(exp_eigvals))
        return res  # Enforce Tr(gamma) = N

    # Step 4: Optimize beta to enforce the trace constraint
    result = minimize(entropy_loss, x0=[0.0], method="Nelder-Mead", options={"maxiter": max_iter, "xatol": tol})
    print(result)
    beta_opt = result.x[0]
    print("beta_opt:", beta_opt)

    # Step 5: Compute refined eigenvalues
    refined_eigvals = np.exp(-beta_opt * eigvals)
    refined_eigvals *= N / np.sum(refined_eigvals)  # Normalize to enforce Tr(gamma) = N

    # Step 6: Reconstruct the refined 1PDM
    gamma_refined = eigvecs @ np.diag(refined_eigvals) @ eigvecs.T.conj()

    return gamma_refined



if __name__ == "__main__":
    data = np.loadtxt("L50/results_L50_nm_1.0_s1k")

    ndim = 4
    N = 2
    dm = np.zeros((ndim, ndim))
    icount = 0
    for i in range(ndim):
        for j in range(i, ndim):
            dm[i, j] = data[icount, 0] - data[icount, 1] + np.mean(data[icount, 2:])
            if i != j:
                dm[j, i] = dm[i, j]
            icount += 1


    print("Noisy 1PDM:\n", repr(np.round(dm, decimals=4)))

    # dm_refined = refine_1PDM_maxent(dm, N)
    # print("Refined 1PDM:\n", repr(np.round(dm_refined, decimals=4)))
