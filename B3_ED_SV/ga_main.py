import numpy as np
import convenience_routines as cr
import scipy
from scipy import optimize, integrate
from scipy.linalg import eigh, expm, det
import ed_solver as ed
# import fci_solver as fci
import sys
import os
import matplotlib
from pylab import *
from functools import reduce
import itertools as it
import sys
from lattice import *
import h5py

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


################### Class for the Anderson Lattice Model (ALM) ###################

class GA:
    def __init__(self, U, nghost, nphysorb, T=0.0002, lcanonical=True, n=0.5, tolconv=1e-7, eks=0):
        self.nphysorb = nphysorb                   # Physical spin orbitals
        self.nghost = nghost
        # Spin quasi orbs (Physical + Ghost)
        self.nquasiorb = nphysorb + nghost
        # Spatial quasi orbs (Physical + Ghost)
        self.nqspo = self.nquasiorb//2

        self.U = U                          # Interaction potential
        self.T = T                          # Fermi temperature
        self.lcanonical = lcanonical
        # If F_i>tolconv, print warning.
        self.tolconv = tolconv
        self.mu = 0.0
        self.eks = np.real(eks)                      # non-local hopping energy

        # Initialize with silly values
        self.n = n if self.lcanonical else -1

        self.H_list = cr.generate_orthonormal_basis(3)

        self.imp_solver = ed.edSolver(self.nphysorb, nghost, 0, 's')

        self.docc = 0.
        self.E1loc = 0.
        self.E2loc = 0.
        self.Eqp = 0.
        self.Etot = 0.

        self.niter = 0  # Number of iterations
        self.ned = 0  # Number of EDs

        self.dos = self.dos_sc
        if lksum:
            self.dos = lambda x: 1.0

    def dos_sc(self, x):
        # Semicircular density of states
        D = (2.0/np.pi)*np.sqrt(1.0-x**(2.0))
        if np.abs(x) > 1:
            D = 0.0
        return D

    def root_mu_Hemb(self, mu):
        self.mu_eh = mu[0]
        self.Lmbdac = self.Lmbdac_orig - np.eye(3)*mu

        self.solve_Hemb()
        self.mu_list.append(mu[0])
        self.muiter += 1

        Trace = self.imp_solver.nc/2.
        if nghost == 2:
            Trace -= 1

        self.microitF3 += 1
        print(
            f"    F3 Micro-It. {self.microitF3}: F3={ Trace-self.n} with self.n: {self.n}")

        return (Trace-self.n)

    def calc_mu_Hemb(self):
        # Determine chemical of Hemb for a given filliing
        self.mu_list = []
        self.muiter = 0
        self.Lmbdac_orig = np.copy(self.Lmbdac+np.eye(self.nqspo)*self.mu_eh)

        chem_pot = optimize.root(self.root_mu_Hemb, self.mu)

        return chem_pot.x

    def calc_Lmbdac(self, Lmbda, Delta):
        # Compute Lagrange multipliers Lambda^c
        lmbda = cr.inverse_realHcombination(Lmbda, self.H_list)
        lmbdac = np.zeros((len(lmbda)))
        DxR = np.dot(self.D, self.R.T)

        for imat in range(len(lmbda)):
            deriv_Delta = cr.dF(
                Delta, self.H_list[imat].T, cr.denRm1, cr.ddenRm1)
            deriv_DeltaxDR = np.trace(np.dot(DxR, deriv_Delta))
            lmbdac[imat] = -lmbda[imat] - \
                (deriv_DeltaxDR + np.conjugate(deriv_DeltaxDR))

        return cr.realHcombination(lmbdac, self.H_list)

    def solve_Hemb(self):
        # H1 = np.array( [[-U/2.]] )

        D, Lmbdac, phasemat, permmat, transmat = self.fix_gauge(
            self.D, self.Lmbdac, lfor_D=True, lreturn_mats=True)
        X = np.diag(Lmbdac).tolist() + D.flatten().tolist()

        self.imp_solver.solve_Hemb(X, self.U)

        # Transform <cd> and <dd> (actually <cg> and <gg>) back to original basis (Transformation order is crucial)
        # Transform back using perutation matrix
        self.imp_solver.fdaggerc = np.dot(permmat.T, self.imp_solver.fdaggerc)
        self.imp_solver.ffdagger = np.dot(
            permmat.T, np.dot(self.imp_solver.ffdagger, permmat))
        # Transform back using matrix of phase factors
        self.imp_solver.fdaggerc = np.dot(phasemat, self.imp_solver.fdaggerc)
        self.imp_solver.ffdagger = np.dot(
            phasemat, np.dot(self.imp_solver.ffdagger, phasemat.T))
        # Transform back using eigenbasis of Lambda^c
        self.imp_solver.fdaggerc = np.dot(transmat, self.imp_solver.fdaggerc)
        self.imp_solver.ffdagger = np.dot(
            transmat, np.dot(self.imp_solver.ffdagger, transmat.T))
        self.ned += 1
        # print(U, self.U)

        # record
        path = f"iter_{self.niter-1}/u_{self._U:.2f}"

        with h5py.File("hembed_cyc1_list.h5", "a") as f:
            if f"{path}/permmat" not in f:
                f[f"{path}/permmat"] = permmat
                f[f"{path}/phasemat"] = phasemat
                f[f"{path}/transmat"] = transmat

    def fix_gauge(self, R, Lmbda, lfor_D=False, lreturn_mats=False):
        """
           Transform R and Lambda to basis where Lambda is diagonal and R_1 > R_2 > R_3 >= 0

           Args:
             R and Lambda in original basis

           return:
             R and Lambda in eigenbasis of Lambda and potentially resorted and with fixed phase
        """

        transmat = scipy.linalg.eigh(Lmbda)[1]
        Lmbda = np.dot(transmat.T, np.dot(Lmbda, transmat))
        R = np.dot(transmat.T, R)

        d_fac = 1. if not lfor_D else -1.

        # Check if R has any negative elements and make them positive and the same for phasemat
        # which transforms <cd> and <dd> according to positive entries in R
        phasemat = np.eye(self.nqspo)
        if d_fac*R[0, 0] < 0.0:
            R[0, 0] *= -1.0
            phasemat[0, 0] *= -1.0

        if self.nqspo > 1:
            if d_fac*R[1, 0] < 0.0:
                R[1, 0] *= -1.0
                phasemat[1, 1] *= -1.0
            if d_fac*R[2, 0] < 0.0:
                R[2, 0] *= -1.0
                phasemat[2, 2] *= -1.0
            Lmbda = np.dot(phasemat, np.dot(Lmbda, phasemat.T))

        # Get the indices of the elements of R in descending order
        idx = np.argsort(-R[:, 0])
        # Create permutation matrix
        permmat = np.zeros((self.nqspo, self.nqspo))
        for i in range(self.nqspo):
            permmat[i, idx[i]] = 1

        # Transform D and Lambda^c by permutation matrix
        R = np.dot(permmat, R)
        Lmbda = np.dot(permmat, np.dot(Lmbda, permmat.T))

        if locdbg:
            print(f"transmat\n{transmat}")
            print(f"phasemat\n{phasemat}")
            print(f"permmat\n{permmat}")
            print(f"Lmbda1 in fix\n{Lmbda}")
            print(f"R1 in fix\n{R}")

        if not lreturn_mats:
            return R, Lmbda
        else:
            return R, Lmbda, phasemat, permmat, transmat

    def optimize_selfc(self, rinit=None, lambdainit=None, muinit=None):
        """
          Explcitly self-consistent way of solving GA equations:

          Internally optimize R, Lambda, mu in each iteration and check convergence based on
          differences between initial and optimized R and Lambda (or Z and total energy)
        """

        def root_GA(xinit):
            tolscRL = 1e-3     # Convergence threshold for R and Lambda
            tolscE = 1e-4      # Convergence threshold for total energy
            tolscZ = 1e-4      # Convergence threshold for Z
            # "RL": Use R and Lambda for convergence check; "EZ": Use total energy and Z
            useconv = "RL"

            diff = 99.9        # Initialize diff with unreasonably large number
            lmu_Hemb = True    # Chemical potential in Hemb or Hqp?
            maxiter = 30       # Maximum number of iterations

            self.mu_fermi = 0.0   # Chemical potential on Hqp
            self.mu = xinit[-1]  # Global chemical potential

            self.lconv = False      # Convergence flag

            while not self.lconv:
                self.niter += 1
                print("####### ITERATION ", self.niter, "#######")

                lmbda0 = xinit[0:self.nqspo*(self.nqspo+1)//2]
                r0 = xinit[self.nqspo*(self.nqspo+1) //
                           2:self.nqspo*(self.nqspo+1)//2 + self.nqspo]

                # Construct R and Lambda from vector
                self.R = np.reshape(np.array(r0), (self.nqspo, 1))
                self.Lmbda = cr.realHcombination(lmbda0, self.H_list)

                # Save R and Lambda separately for convergence check
                R_orig = np.copy(self.R)
                Lmbda_orig = np.copy(self.Lmbda)

                # Fix gauge of R and Lambda
                R_orig, Lmbda_orig = self.fix_gauge(R_orig, Lmbda_orig)

                if self.niter == 1:
                    print("Initial mu", self.mu)
                    print("Initial R")
                    print(self.R)
                    print("Initial Lambda")
                    print(self.Lmbda)
                    print(lmbda0)

                if lmu_Hemb:
                    self.mu_fermi = 0.0
                    self.mu_eh = self.mu
                else:
                    self.mu_fermi = self.mu = self.calc_mu()
                    self.mu_eh = 0.0

                # Compute Delta
                self.calc_Delta()
                self.Delta = self.Delta[0:self.nqspo, 0:self.nqspo]

                # Compute Lagrange multipliers D
                self.calc_D()

                # Compute Lagrange multipliers Lambda^c
                self.Lmbdac = self.calc_Lmbdac(self.Lmbda, self.Delta)

                # Save Z and Etot for convergence check
                self.Z_old = self.calc_Z(Lmbda_orig, R_orig)
                self.Etot_old = np.copy(self.Etot)

                # Diagonalize embeeding Hamiltonian
                self.solve_Hemb()

                # If number of particles in Hemb not what we want: Modify mu
                if not lmu_sweep:
                    if np.abs(self.imp_solver.nc-self.n) > 1e-3:
                        self.microitF3 = 0
                        self.mu = self.calc_mu_Hemb()
                        print(
                            f"F3 Microiterations completed in {self.microitF3} iterations")

                if locdbg:
                    print('n, nc, nu', self.n, self.imp_solver.nc, self.nu)
                    print('mu, mu_fermi, mu_eh', self.mu,
                          self.mu_fermi, self.mu_eh)

                # Try minimizing, if not successful break current loop or continue if F1, F2 sufficiently small (< 1e-5)
                # Minimize F1, F2 with respect to R and Lambda using least-squares minimizer
                uvec = np.hstack((cr.inverse_realHcombination(
                    self.Lmbda, self.H_list), self.R[:, 0]))
                self.microitF1F2 = 0
                try:
                    result = optimize.least_squares(self.cost_func, uvec)
                    print(
                        f"F1, F2 Microiterations converged in {self.microitF1F2} iterations")
                except:
                    if self.F1_max > 1e-5 and self.F2_max > 1e-5:
                        self.R_new = np.copy(R_orig)
                        self.Lmbda_new = np.copy(Lmbda_orig)
                        self.Z = self.calc_Z(Lmbda_orig, R_orig)
                        print(
                            f"F1, F2 Microiterations not converged in 7000 iterations and not below threshold: Stop calculation")
                        break
                    else:
                        print(
                            f"F1, F2 Microiterations not converged in iterations but below thresold: Continue calculation")
                        result = self.xnosol

                if locdbg:
                    print(
                        f'Delta after  microit\n {self.Delta[0:self.nqspo,0:self.nqspo]}')
                    print(f'cd after    microit\n {self.imp_solver.fdaggerc}')
                    print(f'dd after    microit\n {self.imp_solver.ffdagger}')
                    print(f'R after     microit\n {self.R}')
                    print(f'Lambda after microit\n {self.Lmbda}')

                # Construct new R and Lambda from optimized solution
                lmbda0_new = result.x[0:self.nqspo*(self.nqspo+1)//2]
                r0_new = result.x[self.nqspo*(self.nqspo+1) //
                                  2:self.nqspo*(self.nqspo+1)//2 + self.nqspo]
                self.R_new = np.reshape(np.array(r0_new), (self.nqspo, 1))
                self.Lmbda_new = cr.realHcombination(lmbda0_new, self.H_list)

                # Also fix gauge of new R and Lambda
                self.R_new, self.Lmbda_new = self.fix_gauge(
                    self.R_new, self.Lmbda_new)

                # Compute quasi-particle energy
                self.calc_Eqp()

                # Explicit anti-symmetrization of Lambda in Mott phase (only N=1)
                if nghost > 0 and (self.n == 0.5 or (self.mu == 0. and lmu_sweep)):
                    if np.abs(self.Lmbda_new[0, 0]) > np.abs(self.Lmbda_new[2, 2]) and np.abs(self.Lmbda_new[1, 1]) > np.abs(self.Lmbda_new[2, 2]):
                        l1 = (self.Lmbda_new[0, 0] - self.Lmbda_new[1, 1])/2.
                        l2 = (self.Lmbda_new[1, 1] - self.Lmbda_new[0, 0])/2.
                        self.Lmbda_new[0, 0] = l1
                        self.Lmbda_new[1, 1] = l2
                    elif np.abs(self.Lmbda_new[1, 1]) > np.abs(self.Lmbda_new[0, 0]) and np.abs(self.Lmbda_new[2, 2]) > np.abs(self.Lmbda_new[0, 0]):
                        l1 = (self.Lmbda_new[1, 1] - self.Lmbda_new[2, 2])/2.
                        l2 = (self.Lmbda_new[2, 2] - self.Lmbda_new[1, 1])/2.
                        self.Lmbda_new[1, 1] = l1
                        self.Lmbda_new[2, 2] = l2

                # Decompose old and new R and Lambda into basis vectors
                x_new = np.hstack((cr.inverse_realHcombination(
                    self.Lmbda_new, self.H_list), self.R_new[:, 0]))
                x_old = np.hstack((cr.inverse_realHcombination(
                    Lmbda_orig, self.H_list), R_orig[:, 0]))

                # Compute error/difference between old and new R and Lambda for convergence check
                error = np.abs(np.array(x_new)) - np.abs(np.array(x_old))
                RLdiff = np.abs(error).max()
                lconvRL = RLdiff < tolscRL

                # Compute differences in Z and Etot for alternative convergence check
                self.Z = self.calc_Z(self.Lmbda_new, self.R_new)
                self.E1loc = (self.U/2.0)*self.imp_solver.nc
                self.Etot = self.Eqp + self.U*self.docc - self.E1loc + self.U/2.0
                Ediff = np.abs(self.Etot_old-self.Etot)
                lconvE = Ediff < tolscE
                lconvZ = np.abs(self.Z_old-self.Z) < tolscZ

                # Converged?
                self.lconv = (lconvE and lconvZ and useconv == "EZ") or (
                    lconvRL and useconv == "RL")

                Rdiff = np.abs(np.abs(self.R_new) - np.abs(R_orig))
                Ldiff = np.abs(np.abs(self.Lmbda_new) - np.abs(Lmbda_orig))

                if locdbg:
                    print(f"Lambda new\n{self.Lmbda_new}")
                    print(f"R new\n{self.R_new}")
                    print(f"Lambda orig\n{Lmbda_orig}")
                    print(f"R orig\n{R_orig}")
                    print(f"Lambda diff\n{Ldiff}")
                    print(f"R diff\n{Rdiff}")
                    print("mu: ", self.mu, self.mu_eh, self.mu_fermi)
                    print(f"RLdiff", RLdiff)
                    print("Zdiff", np.abs(self.Z_old-self.Z), self.Z_old, self.Z)
                    print(f"Nc: {self.imp_solver.nc}")
                    print(f"Max. error {self.niter},   {RLdiff}")
                    print(f"Z in main {self.Z}")
                print(f"Max. error {self.niter},   {RLdiff}")

                if self.niter > maxiter:
                    break

                xinit = x_new

            self.R = self.R_new
            self.Lmbda = self.Lmbda_new

        self.rinit = rinit
        self.lambdainit = lambdainit
        xinit = np.hstack((lambdainit, rinit, muinit))

        # Call GA solver and find saddle point of Lagrangian
        root_GA(xinit)

        # Check if everything is converged
        self.lconverged_root = True
        if self.F1nrm > self.tolconv:
            print("WARNING: F1 not converged", self.F1nrm)
            self.lconverged_root = False
        if self.F2nrm > self.tolconv:
            print("WARNING: F2 not converged", self.F2nrm)
            self.lconverged_root = False
        self.lconverged_root = self.lconverged_root and self.lconv

    def cost_func(self, xinit):
        lmbda0 = xinit[0:self.nqspo*(self.nqspo+1)//2]
        r0 = xinit[self.nqspo*(self.nqspo+1)//2:self.nqspo *
                   (self.nqspo+1)//2 + self.nqspo]

        self.R = np.reshape(np.array(r0), (self.nqspo, 1))
        self.Lmbda = cr.realHcombination(lmbda0, self.H_list)

        self.calc_Delta()
        self.Delta = self.Delta[0:self.nqspo, 0:self.nqspo]

        if locdbg:
            print(f'<cd>    in microit \n{self.imp_solver.fdaggerc.T}')
            print(f'<dd>    in microit \n{self.imp_solver.ffdagger}')
            print(f'R        in microit \n{self.R}')
            print(f'Lambda   in microit \n{self.Lmbda}')
            print(f'Delta    in microit \n{self.Delta}')
            print(f'Trace    in microit \n{np.trace(self.Delta)-1}')
            print(f'mu_fermi in microit \n{self.mu_fermi}')

        # Compute F1 and F2
        self.F1 = self.imp_solver.fdaggerc.T - \
            np.dot(self.R.T, cr.funcMat(self.Delta, cr.denRm1))
        self.F2 = self.imp_solver.ffdagger - self.Delta

        self.F1_max = np.abs(self.F1).max()
        self.F2_max = np.abs(self.F2).max()

        # Compute norm of F1 and F2 for convergence check
        self.F1nrm = np.linalg.norm(np.power(self.F1, 1), 'fro')
        self.F2nrm = np.linalg.norm(np.power(self.F2, 1), 'fro')

        if locdbg:
            print(f'F1    in microit\n{self.F1}')
            print(f'F2    in microit\n{self.F2}')

        if self.microitF1F2 > 6000:
            self.xnosol = type("X", (object,), {'x': xinit})()
            # Return non-sense such that the solver crashes and we can catch that using an exception
            return ("la")

        self.microitF1F2 += 1
        print(
            f"    F1, F2 Micro-It. {self.microitF1F2}: F1, F2= ", self.F1_max, self. F2_max)
        # sys.exit("LA fct")

        return np.hstack((cr.inverse_realHcombination(self.F2, self.H_list), self.F1[0]))


# mf new Yongxin solver


    def optimize_selfc_new(self, rinit=None, lambdainit=None, muinit=None):
        """
          Explcitly self-consistent way of solving GA equations:

          Internally optimize R, Lambda, mu in each iteration and check convergence based on
          differences between initial and optimized R and Lambda (or Z and total energy)
        """

        def root_GA(xinit):
            tolscRL = 1e-3     # Convergence threshold for R and Lambda
            tolscE = 1e-4      # Convergence threshold for total energy
            tolscZ = 1e-4      # Convergence threshold for Z
            # "RL": Use R and Lambda for convergence check; "EZ": Use total energy and Z
            useconv = "RL"

            diff = 99.9        # Initialize diff with unreasonably large number
            maxiter = 30       # Maximum number of iterations

            self.mu_fermi = 0.0  # Chemical potential on Hqp
            # 0.28987805 #0.0      # Global chemical potential
            self.mu = xinit[-1]
            lmu_Hemb = True

            self.lconv = False      # Convergence flag

            while not self.lconv:
                self.niter += 1
                print("####### ITERATION ", self.niter, "#######")

                lmbda0 = xinit[0:self.nqspo*(self.nqspo+1)//2]
                r0 = xinit[self.nqspo*(self.nqspo+1) //
                           2:self.nqspo*(self.nqspo+1)//2 + self.nqspo]

                # Construct R and Lambda from vector
                self.R = np.reshape(np.array(r0), (self.nqspo, 1))
                self.Lmbda = cr.realHcombination(lmbda0, self.H_list)

                # Save R and Lambda separately for convergence check
                R_orig = np.copy(self.R)
                Lmbda_orig = np.copy(self.Lmbda)

                # Fix gauge of R and Lambda
                R_orig, Lmbda_orig = self.fix_gauge(R_orig, Lmbda_orig)

                if self.niter == 1:
                    print("Initial mu", self.mu)
                    print("Initial R")
                    print(self.R)
                    print("Initial Lambda")
                    print(self.Lmbda)
                    print(lmbda0)

                if lmu_Hemb:
                    self.mu_fermi = 0.0
                    self.mu_eh = self.mu
                else:
                    self.mu_fermi = self.mu = self.calc_mu()
                    self.mu_eh = 0.0

                # Compute Delta
                self.calc_Delta()
                self.Delta = self.Delta[0:self.nqspo, 0:self.nqspo]

                # Compute Lagrange multipliers D
                self.calc_D()

                # Compute Lagrange multipliers Lambda^c
                self.Lmbdac = self.calc_Lmbdac(self.Lmbda, self.Delta)

                # Save Z and Etot for convergence check
                self.Z_old = self.calc_Z(Lmbda_orig, R_orig)
                self.Etot_old = np.copy(self.Etot)

                # Diagonalize embeeding Hamiltonian
                self.solve_Hemb()

                # If number of particles in Hemb not what we want: Modify mu
                if not lmu_sweep:
                    if np.abs(self.imp_solver.nc-self.n) > 1e-3:
                        self.microitF3 = 0
                        self.mu = self.calc_mu_Hemb()
                        print(
                            f"F3 Microiterations completed in {self.microitF3} iterations")

                self.R = (self.imp_solver.fdaggerc.T.dot(
                    cr.funcMat(self.imp_solver.ffdagger, cr.denR))).T
                self.R_new = np.copy(self.R)
#                self.Delta = np.copy(self.imp_solver.ffdagger)

                if locdbg:
                    print('n, nc, nu', self.n, self.imp_solver.nc, self.nu)
                    print('mu, mu_fermi, mu_eh', self.mu,
                          self.mu_fermi, self.mu_eh)

                # Minimize F1, F2 with respect to R and Lambda using least-squares minimizer
                # ,self.mu))
                uvec = np.hstack(
                    (cr.inverse_realHcombination(self.Lmbda, self.H_list)))
                self.microitF1F2 = 0

                # Try minimizing, if not successful break current loop or continue if F1, F2 sufficiently small (< 1e-5)
                result = optimize.least_squares(self.cost_func_Delta, uvec)

                # record
                path = f"iter_{self.niter-1}/u_{self._U:.2f}"

                with h5py.File("hembed_cyc1_list.h5", "a") as f:
                    if f"{path}/lambda" not in f:
                        f[f"{path}/lambda"] = self.Lmbda

                print(
                    f"F1, F2 Microiterations converged in {self.microitF1F2} iterations")

#                print(result.x)
#                sys.exit()

                if locdbg:
                    print(
                        f'Delta after  microit\n {self.Delta[0:self.nqspo,0:self.nqspo]}')
                    print(f'cd after    microit\n {self.imp_solver.fdaggerc}')
                    print(f'dd after    microit\n {self.imp_solver.ffdagger}')
                    print(f'R after     microit\n {self.R}')
                    print(f'Lambda after microit\n {self.Lmbda}')

                # Construct new R and Lambda from optimized solution
                lmbda0_new = result.x[0:self.nqspo*(self.nqspo+1)//2]
                self.Lmbda_new = cr.realHcombination(lmbda0_new, self.H_list)
#                self.mu = 0. #result.x[-1]

                # Also fix gauge of new R and Lambda
                self.R_new, self.Lmbda_new = self.fix_gauge(
                    self.R_new, self.Lmbda_new)

                # Compute quasi-particle energy
                self.calc_Eqp()

                # Explicit anti-symmetrization of Lambda in Mott phase (only N=1)
                if nghost > 0 and (self.n == 0.5 or (self.mu == 0. and lmu_sweep)):
                    if np.abs(self.Lmbda_new[0, 0]) > np.abs(self.Lmbda_new[2, 2]) and np.abs(self.Lmbda_new[1, 1]) > np.abs(self.Lmbda_new[2, 2]):
                        l1 = (self.Lmbda_new[0, 0] - self.Lmbda_new[1, 1])/2.
                        l2 = (self.Lmbda_new[1, 1] - self.Lmbda_new[0, 0])/2.
                        self.Lmbda_new[0, 0] = l1
                        self.Lmbda_new[1, 1] = l2
                    elif np.abs(self.Lmbda_new[1, 1]) > np.abs(self.Lmbda_new[0, 0]) and np.abs(self.Lmbda_new[2, 2]) > np.abs(self.Lmbda_new[0, 0]):
                        l1 = (self.Lmbda_new[1, 1] - self.Lmbda_new[2, 2])/2.
                        l2 = (self.Lmbda_new[2, 2] - self.Lmbda_new[1, 1])/2.
                        self.Lmbda_new[1, 1] = l1
                        self.Lmbda_new[2, 2] = l2

                # Decompose old and new R and Lambda into basis vectors
                x_new = np.hstack((cr.inverse_realHcombination(
                    self.Lmbda_new, self.H_list), self.R_new[:, 0]))
                x_old = np.hstack((cr.inverse_realHcombination(
                    Lmbda_orig, self.H_list), R_orig[:, 0]))

                # Compute error/difference between old and new R and Lambda for convergence check
                error = np.abs(np.array(x_new)) - np.abs(np.array(x_old))
                RLdiff = np.abs(error).max()
                lconvRL = RLdiff < tolscRL

                # Compute differences in Z and Etot for alternative convergence check
                self.Z = self.calc_Z(self.Lmbda_new, self.R_new)
                self.E1loc = (self.U/2.0)*self.imp_solver.nc
                self.Etot = self.Eqp + self.U*self.docc - self.E1loc + self.U/2.0
                Ediff = np.abs(self.Etot_old-self.Etot)
                lconvE = Ediff < tolscE
                lconvZ = np.abs(self.Z_old-self.Z) < tolscZ

                # Converged?
                self.lconv = (lconvE and lconvZ and useconv == "EZ") or (
                    lconvRL and useconv == "RL")

                Rdiff = np.abs(np.abs(self.R_new) - np.abs(R_orig))
                Ldiff = np.abs(np.abs(self.Lmbda_new) - np.abs(Lmbda_orig))

                # if locdbg:
                if True:
                    print("U:", self.U)
                    print(f"Lambda new\n{self.Lmbda_new}")
                    print(f"R new\n{self.R_new}")
                    print(f"Lambda orig\n{Lmbda_orig}")
                    print(f"R orig\n{R_orig}")
                    print(f"Lambda diff\n{Ldiff}")
                    print(f"R diff\n{Rdiff}")
                    print("mu: ", self.mu, self.mu_eh, self.mu_fermi)
                    print(f"RLdiff", RLdiff)
                    print("Zdiff", np.abs(self.Z_old-self.Z), self.Z_old, self.Z)
                    print(f"Nc: {self.imp_solver.nc}")
                    print(f"Max. error {self.niter},   {RLdiff}")
                    print(f"Z in main {self.Z}")
                print(f"Max. error {self.niter},   {RLdiff}")

                if self.niter > maxiter:
                    break

                xinit = x_new

            self.R = self.R_new
            self.Lmbda = self.Lmbda_new

        self.rinit = rinit
        self.lambdainit = lambdainit
        xinit = np.hstack((lambdainit, rinit, muinit))

        # Call GA solver and find saddle point of Lagrangian
        root_GA(xinit)

        # Check if everything is converged
        self.lconverged_root = True
        if self.F1nrm > self.tolconv:
            print("WARNING: F1 not converged", self.F1nrm)
            self.lconverged_root = False
        if self.F2nrm > self.tolconv:
            print("WARNING: F2 not converged", self.F2nrm)
            self.lconverged_root = False
        self.lconverged_root = self.lconverged_root and self.lconv

    def cost_func_Delta(self, xinit):
        lmbda0 = xinit[0:self.nqspo*(self.nqspo+1)//2]
#           self.mu_fermi = 0. #xinit[-1]

        self.Lmbda = cr.realHcombination(lmbda0, self.H_list)

        self.calc_Delta()
        self.Delta = self.Delta[0:self.nqspo, 0:self.nqspo]

        locdbg = True
        if locdbg:
            print(f'<cd>    in microit \n{self.imp_solver.fdaggerc.T}')
            print(f'<dd>    in microit \n{self.imp_solver.ffdagger}')
            print(f'R        in microit \n{self.R}')
            print(f'Lambda   in microit \n{self.Lmbda}')
            print(f'Delta    in microit \n{self.Delta}')
            print(f'Trace    in microit \n{np.trace(self.Delta)-1}')
            print(f'mu_fermi in microit \n{self.mu_fermi}')

        # Compute F1 and F2
        self.F1 = self.n - (np.trace(self.Delta)-1)
        self.F2 = self.imp_solver.ffdagger - self.Delta

        self.F1_max = np.abs(self.F1).max()
        self.F2_max = np.abs(self.F2).max()

        # Compute norm of F1 and F2 for convergence check
        self.F1nrm = self.F1_max  # np.linalg.norm(np.power(self.F1,1),'fro')
        self.F2nrm = np.linalg.norm(np.power(self.F2, 1), 'fro')

        if locdbg:
            print(f'F1    in microit\n{self.F1}')
            print(f'F2    in microit\n{self.F2}')

        self.microitF1F2 += 1
        print(
            f"    F1, F2 Micro-It. {self.microitF1F2}: F1, F2= ", self.F1_max, self. F2_max)
        print("with xinit:", xinit, "dim:", self.nqspo)

        # ,self.F1))
        err = np.hstack((cr.inverse_realHcombination(self.F2, self.H_list)))
        print("err:", err)

        return err


# mf new Yongxin solver


    def calc_Z(self, Lmbda, R):
        # Compute quasi-particle weight Z
        if self.nqspo > 1:
            R_orig, Lmbda_orig = self.fix_gauge(R, Lmbda)

            if locdbg:
                print(f"R in Z: \n{R_orig}")
                print(f"Lmbda in Z: \n{Lmbda_orig}")

            l1 = Lmbda_orig[0, 0] - self.mu_fermi
            l2 = Lmbda_orig[1, 1] - self.mu_fermi
            l3 = Lmbda_orig[2, 2] - self.mu_fermi

            r1 = R_orig[0, 0]
            r2 = R_orig[1, 0]
            r3 = R_orig[2, 0]

            return (l2*l3*r1**2 + l1*l3*r2**2 + l1*l2*r3**2)**2/(l1**2*l3**2*r2**2 + l2**2*(l3**2*r1**2 + l1**2*r3**2))
        else:
            return R[0, 0]*R[0, 0]

    def make_Hqp(self, x):
        Roprod = np.dot(self.R, self.R.T)
        # R_i1 x R^\dag_1j

        # Hqp      = np.zeros((self.nqspo,self.nqspo))
        Hqp = Roprod*x*alpha + self.Lmbda - self.mu_fermi*np.eye(self.nqspo)

        return Hqp

    # Compute uncorrelated one-electron density Delta

    def calc_Delta(self):
        def integrand(x, idx, jdx):
            Hqp = self.make_Hqp(x)

            f = self.dos(x)*cr.calc_C(Hqp, T=self.T)[idx, jdx]

            return f

        Roprod = np.zeros((self.nqspo, self.nqspo))
        Roprod[0:self.nqspo, 0:self.nqspo] = np.dot(self.R, self.R.T)

        Delta = np.zeros((self.nqspo, self.nqspo))

        if lksum:
            Roprod = np.dot(self.R, self.R.T)
            Hqp_tmp = self.Lmbda - self.mu_fermi*np.eye(self.nqspo)
            for ek in self.eks:
                Hqp = Roprod*ek*alpha + Hqp_tmp
                f = cr.calc_C(Hqp, T=self.T)
                for i in range(self.nqspo):
                    for j in range(i, self.nqspo):
                        Delta[i, j] += f[i, j]

            for i in range(self.nqspo):
                for j in range(i, self.nqspo):
                    Delta[j, i] = Delta[i, j]
            Delta /= len(self.eks)
        else:
            for i in range(self.nqspo):
                for j in range(i, self.nqspo):

                    if lksum:
                        integral = 0.0
                        for ek in self.eks:
                            integral += integrand(ek, i, j)/len(self.eks)
                    else:
                        integral, error = integrate.quad(
                            integrand, -1, 1, args=(i, j), epsabs=1e-14)

                    Delta[i, j] = integral
                    Delta[j, i] = integral
        self.Delta = Delta
        self.nu = 0.0   # In the Hubbard model we do not have an uncorrelated band

        # print("\nlam:", self.Lmbda)
        # print("R:",  self.R)
        # print(Delta)
        # print("calc Delta done."); input()

    # Compute Lagrange multipliers D

    def calc_D(self):
        def integrand(x, idx):
            Hk = np.zeros((self.nqspo, self.nqspo))
            Hqp = np.zeros((self.nqspo, self.nqspo))
            Rt = np.zeros((self.nqspo, self.nqspo))
            Lambdat = np.zeros((self.nqspo, self.nqspo))

            Rt[0:self.nqspo, 0] = np.copy(self.R[:, 0])

            # Set up Hk
            Hk[0, 0] = alpha*x

            # Compute Hqp
            Hqp = self.make_Hqp(x)

            resl = Hk.dot(Rt.T)
            resr = cr.calc_C(Hqp, T=self.T)
            f = self.dos(x)*resl.dot(resr)[0, idx]

            return f

        Left = np.zeros((self.nqspo))
        for i in range(self.nqspo):

            if lksum:
                integral = 0.0
                for ek in self.eks:
                    integral += integrand(ek, i)/len(self.eks)
            else:
                integral, error = integrate.quad(
                    integrand, -1, 1, args=(i,), epsabs=1e-14)

            Left[i] = integral

        Left = np.reshape(Left, (self.nqspo, 1))
        Right = cr.funcMat(self.Delta, cr.denR)

        self.D = np.dot(Right, Left)

    def calc_Eqp(self):
        # Compute quasi-particle energy
        def integrand(x):
            Hk = np.zeros((self.nqspo, self.nqspo))
            Rt = np.zeros((self.nqspo, self.nqspo))
            Lambdat = np.zeros((self.nqspo, self.nqspo))

            # Set up R-tilde
            Rt[0:self.nqspo, 0] = np.copy(self.R[:, 0])

            # Set up Hk
            Hk[0, 0] = alpha*x

            # Compute Hqp
            Hqp = self.make_Hqp(x)

            resl = Rt.dot(Hk.dot(Rt.T))
            resr = cr.calc_C(Hqp, T=self.T)

            f = self.dos(x)*np.trace(resl.dot(resr))

            return f

        if lksum:
            integral = 0.0
            for ek in self.eks:
                integral += integrand(ek)
            integral /= len(self.eks)
        else:
            integral, error = integrate.quad(integrand, -1, 1, epsabs=1e-14)

        self.Eqp = 2.0*integral


######### Set up tight-binding and lattice to be used ########
clat = "Inf"
lksum = clat != "Inf"
eks = -99

if not lksum:
    print("Use semicircular DOS (infinite lattice)")

# lattice(dimension,3D unit-cell vector, number of k-points in three directions, number of atoms, atom's Cartesian coordinates)
if lksum:

    if clat == "2Dsquare":
        lat = lattice(2, [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                      [30, 30, 1], 1, [[0, 0, 0]])

        # hopping dictionary {atom index:, [[atom index, (cell xindex, cell yindex, cell zindex), hopping]]}
        t = -1.  # hopping amplitude
        hop = {}
        hop[0] = [[0, 1, 0, 0, t],
                  [0, -1, 0, 0, t],
                  [0, 0, 1, 0, t],
                  [0, 0, -1, 0, t]]
    elif clat == "2Dtriangular":
        lat = lattice(2, [[1, 0, 0], [0.5, 1, 0], [0, 0, 0]],
                      [50, 50, 1], 1, [[0, 0, 0]])

        # hopping dictionary {atom index:, [[atom index, (cell xindex, cell yindex, cell zindex), hopping]]}
        t = -1.0
        hop = {}
        hop[0] = [[0, 1, 0, 0, t],
                  [0, 0, 1, 0, t],
                  [0, -1, 1, 0, t],
                  [0, -1, 0, 0, t],
                  [0, 0, -1, 0, t],
                  [0, 1, -1, 0, t]]
    elif clat == "3Dcubic":
        lat = lattice(3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      [16, 16, 16], 1, [[0, 0, 0]])

        # hopping dictionary {atom index:, [[atom index, (cell xindex, cell yindex, cell zindex), hopping]]}
        t = -1.
        hop = {}
        hop[0] = [[0, 1, 0, 0, t],
                  [0, -1, 0, 0, t],
                  [0, 0, 1, 0, t],
                  [0, 0, -1, 0, t],
                  [0, 0, 0, 1, t],
                  [0, 0, 0, -1, t]]

    hop_mat = lat.build_mat_TB_PBC(hop, antiperiodic=False)
    evals, evecs = eigh(hop_mat)

    # Fourier to k-space
    hop_k = lat.FT_hop_ij()
    eks = hop_k.diagonal()


######### Hamiltonan settings #########
nphysorb = 2                 # Number of physical spin orbitals
nghost = 4                 # Number of ghost/auxiliary spin orbitals


#########     ALM settings   #########
# Charge-Transfer energy: xct = eps_d - eps_p := eps_p
shift_list = (-1.0,)
V = 1.0               # Hybridization of correlated and uncorrelated band
# Non-dispersive correlated band => alpha = 0 (i.e. no hopping)
alpha = 0.0
# Hopping in uncorrelated band (Only set to 0 for specific calculations, otherwise default is tij=1)
tij = 1.0
T = 0.002             # Artificial temperature for Fermi-smearing


######## Setting for Hubbard model to make code more stable ########
lhubbard = True                          # Use the Hubbard model or the ALM?
if lhubbard:
    # Hopping between orbitals is set to standard for the Hubbard model
    alpha = 1.0
    # Fermi temperature set t T=0 since it is more stable (Not sure about g-GA)
    T = 0.003

lmicroit_mu = False  # If True, chemical potential is determined in each iteration and not used as a parameter of the root solver


######### Miscalleneous settings ##########
locdbg = False          # Some more output for debugging


print("################### PARAMETERS/SETTINGS OF THE CALCULATION #######################")

ntot_list = (0.5,)
lmu_sweep = np.abs(ntot_list[0]-0.5) < 1e-7

U_start = 2.5
U_end = 2.51
U_step = 0.6

# For calculations in grand-canonical ensemble (didn't check that for quite a while)
mu_start = 0.0
mu_end = 0.0
mu_step = 1.0  # Is a dummy in case mu_start=mu_end


print("U_start, U_end, U_step, U_c", U_start, U_end, U_step)
print("mu_start, mu_end, mu_step", mu_start, mu_end, mu_step)
print("Lambda antisymmetrization", ntot_list[0] == 0.5)


# For half-filling start from Mott phase (irep_start = 1) and perform a forward and backward
# sweep (nrep = 3) to obtain Mott and metallic solutions
irep_start = 1
nrep = 3

# For doped calculations start from metallic phase (irep_start = 2) and only perform a forward sweep
if nghost == 0 or np.abs(ntot_list[0]-0.5) > 1e-7:
    irep_start = 2
    nrep = 3

# Loop over particle numbers
for ntot in ntot_list:
    icnt = 0
    GA_list = []

    # For obtaining Mott and metallic phase (2 repetitions)
    # For dopes calculations, this loop will only be executed once (1 "repetition")
    # for reploop in range(irep_start,nrep):
    for reploop in [2]:

        print(f"------- REPETION LOOP Nr. {reploop} --------")

        # Half-filling
        if reploop % 2 == 0 or nghost == 0:
            # U_list      = np.arange(U_start,U_end+U_step,U_step) # Forward sweep
            # U_list      = np.linspace(0, 5.5, 12) # Forward sweep
            # U_list[0] += 0.1

            U_list = [2.5]

        if reploop % 2 != 0 and nghost > 0:
            # U_list      = np.arange(U_start,U_end+U_step,U_step)[::-1]
            # U_list      = np.linspace(0, 5.5, 12)
            # U_list[0] += 0.1

            U_list = [2.5]
            U_list = U_list[::-1]

        if np.abs(U_start-U_end) <= 1e-7:
            U_list = (U_start,)

        z_list = []
        nc_list = []

        # Loop over interaction strengths
        for U in U_list:

            if reploop % 2 != 0:
                mu_list = np.arange(mu_start, mu_end, mu_step)
            if reploop % 2 == 0:
                mu_list = np.arange(mu_start, mu_end, mu_step)[::-1]
            if np.abs(mu_start-mu_end) <= 1e-7:
                mu_list = (mu_start,)

            # Loop over chemical potentials for grand-canonical calculations
            for mu in mu_list:

                # Only use these initial guesses for very first calculations, afterwards use adiabatic switching
                if icnt == 0 or nghost == 0:

                    ######## For full half-filling sweep #######
                    if reploop % 2 != 0 and nghost > 0 and np.abs(ntot-0.5) < 1e-7:
                        # Hubbard Mott
                        rinit = (0.0, np.sqrt(0.95/2), np.sqrt(0.95/2))
                        lambdainit = (0.0, (U)/2, -(U)/2, 0.0, 0.0, 0.0)
                        # lambdainit = (0,0,0,0.0,0,0.0)
                    else:
                        rinit = (1, 0, 0)
                        # lambdainit = (0.1,0,-0.10,0,0,0)
                        lambdainit = (0, 0, 0, 0.0, 0, 0.0)
                        muinit = 0.0

                    if nghost == 0:
                        rinit = 1.0
                        lambdainit = 0.0

                    if lmu_sweep:
                        muinit = np.copy(mu)
                else:
                    # Adiabatic switching: Use previous solution as initial guess
                    if GA_list[icnt-1].nqspo > 1:
                        rinit = (
                            GA_list[icnt-1].R[0, 0], GA_list[icnt-1].R[1, 0], GA_list[icnt-1].R[2, 0])
                    else:
                        rinit = GA_list[icnt-1].R[0, 0]
                    lambdainit = cr.inverse_realHcombination(
                        GA_list[icnt-1].Lmbda, GA_list[icnt-1].H_list)
                    if not lmu_sweep:
                        muinit = (GA_list[icnt-1].mu)

                try:
                    # load solution
                    with h5py.File("ggasolutions.h5", "r") as f:
                        print(f"cycle_{reploop}/n_{ntot:.1f}/U_{U:.1f}/rinit")

                        rinit = f[f"cycle_{reploop}/n_{ntot:.1f}/U_{U:.1f}/rinit"][()]
                        lambdainit = f[f"cycle_{reploop}/n_{ntot:.1f}/U_{U:.1f}/lambdainit"][()]
                except:
                    pass

                if not lmu_sweep:
                    print("Starting calculation for U, N = ", str(
                        "{:.4f}".format(U)), str("{:.4f}".format(ntot)))
                else:
                    print("Starting calculation for U, mu = ", str(
                        "{:.4f}".format(U)), str("{:.4f}".format(muinit)))

        # Initialize GA oobject
                GA_list.append(GA(U, nghost, nphysorb, n=ntot,
                               T=T, lcanonical=True, tolconv=1e-3, eks=eks))
                if icnt > 0:
                    GA_list[icnt].mu = GA_list[icnt-1].mu

                if lmu_sweep:
                    muinit = mu

                # record
                GA_list[icnt]._U = U
                GA_list[icnt]._reploop = reploop

                # Solve GA equations
                GA_list[icnt].optimize_selfc_new(rinit, lambdainit, muinit)

                # save solution
                with h5py.File("ggasolutions.h5", "a") as f:
                    rinit = (
                        GA_list[icnt-1].R[0, 0], GA_list[icnt-1].R[1, 0], GA_list[icnt-1].R[2, 0])
                    lambdainit = cr.inverse_realHcombination(
                        GA_list[icnt].Lmbda, GA_list[icnt].H_list)
                    if f"cycle_{reploop}/n_{ntot:.1f}/U_{U:.1f}/rinit" not in f:
                        f[f"cycle_{reploop}/n_{ntot:.1f}/U_{U:.1f}/rinit"] = rinit
                        f[f"cycle_{reploop}/n_{ntot:.1f}/U_{U:.1f}/lambdainit"] = lambdainit

                z_list.append(GA_list[icnt].Z)
                nc_list.append(GA_list[icnt].imp_solver.nc)

                # ------- Print Results ------- #
                print("Final mu=", GA_list[icnt].mu)

                Z = GA_list[icnt].R[0, 0]**2  # .T.dot(GA_list[icnt].R)
                if nghost > 0:
                    Z = GA_list[icnt].Z

                if nghost > 0:
                    print("Final Lambda elements", GA_list[icnt].Lmbda[0, 0],
                          GA_list[icnt].Lmbda[1, 1], GA_list[icnt].Lmbda[2, 2])
                    print("Final R elements", GA_list[icnt].R[0, 0],
                          GA_list[icnt].R[1, 0], GA_list[icnt].R[2, 0])
                    print("\n")

                    print("Final Lambda^c elements", GA_list[icnt].Lmbdac[0, 0],
                          GA_list[icnt].Lmbdac[1, 1], GA_list[icnt].Lmbdac[2, 2])
                    print("Final D elements", GA_list[icnt].D[0, 0],
                          GA_list[icnt].D[1, 0], GA_list[icnt].D[2, 0])

                    print("\n")

                print("U:", U, "mu:", mu)
                print("Final Z:", Z)
                print("Final Double occupancy:", GA_list[icnt].imp_solver.docc)
                print("Final n correlated:", GA_list[icnt].imp_solver.nc)
                print("Final Eqp:", GA_list[icnt].Eqp)
                print("Final E2loc:", GA_list[icnt].imp_solver.E2loc)
                print("Final E1loc:", GA_list[icnt].imp_solver.E1loc)
                print("Final Etot:", GA_list[icnt].Etot)
                print("Nr. of iterations", GA_list[icnt].niter)
                print("Nr. of EDs", GA_list[icnt].ned)

                if not lmu_sweep:
                    if GA_list[icnt].lconverged_root:
                        print("GA Converged for U, N = ", str(
                            "{:.4f}".format(U)), str("{:.4f}".format(ntot)))
                    else:
                        print("GA NOT converged for U, N = ", str(
                            "{:.4f}".format(U)), str("{:.4f}".format(ntot)))
                else:
                    if GA_list[icnt].lconverged_root:
                        print("GA Converged for U, mu = ", str(
                            "{:.4f}".format(U)), str("{:.4f}".format(mu)))
                    else:
                        print("GA NOT converged for U, mu = ", str(
                            "{:.4f}".format(U)), str("{:.4f}".format(mu)))

                icnt += 1
