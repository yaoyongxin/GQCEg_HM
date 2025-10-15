import h5py, numpy
import matplotlib.pyplot as plt



def get_rotated_rmat(dm, permmat, phasemat, transmat):
    # accoding to ga_main.py
    fdaggerc = dm[1:, :1]
    cdaggerc = dm[:1, :1]
    ffdagger = numpy.eye(fdaggerc.shape[0]) - dm[1:, 1:]
    fdaggerc = numpy.linalg.multi_dot([transmat, phasemat, permmat.T, fdaggerc])
    ffdagger = numpy.linalg.multi_dot([transmat, phasemat, permmat.T, ffdagger,
            permmat, phasemat.T, transmat.T])

    print("fdaggerc, ffdagger:")
    print(numpy.round(fdaggerc, decimals=3))
    print(numpy.round(ffdagger, decimals=3))

    rmat = fdaggerc.T.dot(funcMat(ffdagger, denR)).T
    print("rmat:")
    print(rmat)
    return rmat


def calc_spectral_function(rmat, lmda, η=0.01, ne=1000):
    emesh = numpy.linspace(-2.2, 2.2, num=500)
    dmesh = 0
    dos = semicircular()
    elist = dos.get_e_list_of_uniform_wt(nmesh=ne)
    # run through each elist point
    roprod = rmat.dot(rmat.T)
    for e in elist:
        hmat = e*roprod + lmda
        evals, evecs = numpy.linalg.eigh(hmat)
        wlist = numpy.einsum("ja,ai,ib,bj->i", rmat.T, evecs, evecs.T, rmat)

        for i, x in enumerate(evals):
            fun = η/((emesh - x)**2 + η**2)
            dmesh += fun*wlist[i]
    dmesh /= numpy.pi*ne

    # plot
    plt.plot(emesh, dmesh)
    plt.show()


def sigma_jax_e(omega, e, rmat, lmda, η=0.0001):
    import jax
    nb, nc = rmat.shape
    rmat = jax.numpy.array(rmat)
    lmda = jax.numpy.array(lmda)
    roprod = rmat.dot(rmat.T)
    hmat = e*roprod + lmda
    gfun = rmat.T.dot(jax.numpy.linalg.inv(
            jax.numpy.eye(nb)*(omega + 1j*η) - hmat)).dot(rmat)

    sfun = jax.numpy.eye(nc)*(omega + 1j*η - e) \
            - jax.numpy.linalg.inv(gfun)
    return sfun


def calc_self_energy(rmat, lmda, η=0.01):
    emesh = numpy.linspace(-2.2, 2.2, num=500)
    slist = []
    e = 0
    for omega in emesh:
        slist.append(sigma_jax_e(omega, e, rmat, lmda, η=η)[0,0])

    print("max imag of self energy:", numpy.max(abs(numpy.asarray(slist).imag)))
    plt.plot(emesh, slist)
    plt.show()

def get_z(e, rmat, lmda):
    from jax import jacfwd
    dSigma_domega = jacfwd(sigma_jax_e, argnums=0)
    omega = 0.
    ds_do = dSigma_domega(omega, e, rmat, lmda)
    zval = 1/(1 - ds_do[0, 0].real)
    print(f"numerical z: {zval}")


class semicircular(object):
    '''semi-circular DOS.
    '''
    def __init__(self):
        '''define dos and cumulative dos function.
        '''
        self.dos = lambda e: 2./numpy.pi * numpy.sqrt(1-e**2)
        self.cdos = lambda e: (e*numpy.sqrt(1-e**2) \
                + numpy.arcsin(e)) / numpy.pi + 0.5

    def get_e_list_of_uniform_wt(self, nmesh=5000):
        '''Get the energy mesh with uniform weight.
        '''
        cdos_list = numpy.linspace(0,1,nmesh+1)
        from scipy.optimize import bisect
        e_list = [bisect(lambda x: self.cdos(x)-a, -1 ,1) \
                for a in cdos_list]
        e_list = numpy.asarray(e_list)
        e_list = (e_list[1:] + e_list[0:-1])/2
        return e_list


#  [D(1-D)]^(-0.5)
def denR(x, tol=1e-5):
    x[numpy.where(x < tol)] = tol
    x[numpy.where(x > 1-tol)] = 1-tol
    return (x*(1-x))**(-0.5)


# calc Z
def calc_z(lam, rmat):
    assert(len(lam) == 3)
    l, v = numpy.linalg.eigh(lam)
    rmat = v.T.dot(rmat)
    l1, l2, l3 = l
    r1, r2, r3 = rmat

    return (l2*l3*r1**2 + l1*l3*r2**2 + l1*l2*r3**2)**2/(l1**2*l3**2*r2**2 + l2**2*(l3**2*r1**2 + l1**2*r3**2))


# Generic function of an Hermitian matrix
def funcMat(H, function, T=0.0, pr=False):
    assert(H.shape[0] == H.shape[1])
    if not numpy.allclose(H.conj().T,H):
      print (H)
    assert(numpy.allclose(H.conj().T,H))

    eigenvalues, U = numpy.linalg.eigh(H)

    print("dm evals:", eigenvalues)
    eigenvalues *= 1.5/sum(eigenvalues)
    print("adjusted dm evals:", eigenvalues)

    if T>0:
        functioneigenvalues = function(eigenvalues,T=T)
    else:
        functioneigenvalues = function(eigenvalues)
    if pr:
        print (functioneigenvalues)

    functionH = numpy.linalg.multi_dot([U, numpy.diag(functioneigenvalues), U.conj().T])
    return functionH



if __name__ == "__main__":
    path = "/iter_13/u_2.50"
    with h5py.File("hembed_cyc2_list.h5", "r") as f:
        dm_ref0 = f[f"{path}/DM"][::2, ::2]
        lam = f[f"{path}/lambda"][()]
        permmat = f[f"{path}/permmat"][()]
        phasemat = f[f"{path}/phasemat"][()]
        transmat = f[f"{path}/transmat"][()]

    with h5py.File("dm.h5", "r") as f:
        dm_ref1 = f["/dm_ref"][()]
        dm = f["/dm"][()]

    print("dm max error:")
    print(abs(dm - dm_ref0).max())

    rmat = get_rotated_rmat(dm, permmat, phasemat, transmat)
