# author: Yongxin Yao (yxphysice@gmail.com)
from mpi4py import MPI
import numpy, h5py, pickle, warnings, itertools
import scipy.optimize, scipy.linalg, scipy.sparse
from timing import timeit
import os, time


class ansatz:
    '''
    define the main procedures of one step of avqite calculation.
    set up: 1) default reference state of the ansatz.
            2) operator pool
    '''
    def __init__(self,
            model,           # the qubit model
            rcut=1.e-2,      # McLachlan distance cut-off
            fcut=1.e-1,      # cut-off ratio for invidual contribution
            max_add=5,       # maximal number of new pauli rotation gates to be added
            maxntheta=-1,    # maximal variational parameters allowed
            bounds=(-5, 5),  # bounds for dtheta/dt
            delta=1.e-4,     # Tikhonov regularization parameter if >= 0,
                             # otherwise switch to lsq_linear
            invmode=1,       # linear equation solver mode.
            dt=0.02,         # time step
            tsave=1200,      # time interval to save intermediate results
            vtol=1e-4,       # gradient cutoff
            tetras=True,     # use tetras by default
            tf=numpy.inf,    # maximal final time it can reach
            psmode=1,        # pool screening mode: 0: uisng gradient only; else: McLachlan distance.
            ):
        self._comm = MPI.COMM_WORLD
        self._mrank = self._comm.Get_rank()
        self._msize = self._comm.Get_size()

        self._model = model
        self._nq = model._nsite
        self._state = None
        self._rcut = rcut
        self._rcut0 = rcut*fcut
        self._max_add = max_add
        self._maxntheta = maxntheta
        self._bounds = bounds
        self._delta = delta
        self._invmode = invmode
        self._dt = dt
        self._tsave = tsave
        self._vtol = vtol
        self._tetras = tetras
        self._tf = tf
        self._psmode = psmode
        # generate operator pool
        self.setup_pool()
        self.init_ansatz()

    def init_ansatz(self):
        self._ansatz = [[], []]
        # set reference state
        self.set_ref_state()
        # variational parameters.
        self._params = []
        # the op indices in pool for the ansatz
        self._ansatz_pids = []
        self._ngates = [0]*self._nq
        self._t = 0
        self._layer_range = []
        self._iter = 0
        self._dthdt = None

        if os.path.isfile("ansatz.pkle"):
            self.pload_ansatz()
        # list of unitaries interms of operators and index labels
        elif os.path.isfile("ansatz_simp.pkle"):
            self.pload_ansatz_simp()
        elif os.path.isfile("ansatz_inp.pkle"):
            self.pload_ansatz_inp()

        if self._mrank == 0:
            print(f"nparams in the initial ansatz: {len(self._ansatz_pids)}")

    @property
    def state(self):
        return self._state

    @property
    def ngates(self):
        return self._ngates[:]

    def update_state(self):
        self._state = self.get_state()

    @timeit
    def setup_pool(self):
        '''
        setup pool from incar.
        '''
        raise NotImplementedError

    def set_ref_state(self):
        '''set reference state from ref_state.inp file.
        '''
        if os.path.isfile("ref_state.npy"):
            self._ref_state = numpy.load("ref_state.npy")
        else:
            ref = self._model._incar["ref_state"]
            # binary literal to int
            nnz = int(ref, 2)
            self._ref_state = numpy.zeros((2**self._nq), dtype=numpy.complex128)
            self._ref_state[nnz] = 1.

    def update_ngates(self):
        '''
        update gate counts.
        '''
        raise NotImplementedError

    def get_dthdt(self, mmat, vvec, rtol=1e-6):
        if self._delta < 0:
            res = scipy.optimize.lsq_linear(mmat,
                    vvec,
                    bounds=self._bounds,
                    # lsq_solver='lsmr',
                    )
            dthdt = res["x"]
        else:
            a = mmat + self._delta*numpy.eye(mmat.shape[0])
            if self._invmode == 0:
                ainv = numpy.linalg.inv(a)
                dthdt = ainv.dot(vvec)
            else:
                # use cg.
                if self._dthdt is None:
                    x0 = None
                elif len(self._dthdt) == len(vvec):
                    x0 = self._dthdt
                else:
                    x0 = numpy.zeros_like(vvec)
                    x0[:len(self._dthdt)] = self._dthdt

                dthdt, info = scipy.sparse.linalg.cg(a, vvec, x0=x0, rtol=rtol)
                assert(info == 0), f"info = {info} from scipy.sparse.linalg.cg."
        return dthdt

    def get_scores(self, inds_chk):
        scores = numpy.zeros(max(inds_chk)+1)

        for i, ind in enumerate(inds_chk):
            if i%self._msize == self._mrank:
                scores[ind], _, _, _, _ = self.get_score([ind], mode=self._psmode)
        scores = self._comm.allreduce(scores)

        scores_shft = -numpy.ones(max(inds_chk)+1)
        scores_shft[numpy.r_[inds_chk]] = self._scores_bias[numpy.r_[inds_chk]]
        scores += scores_shft
        return numpy.asarray(scores)

    def get_score(self, inds, mode=0):
        mmat, vvec, vecp_adds = self.get_mvhelp(inds)
        dthdt = self.get_dthdt(mmat, vvec)
        dist_p = vvec.dot(dthdt)
        score = dist_p.real - self._distp

        if mode == 0:
            score = abs(vvec[-1])

        return score, mmat, vvec, vecp_adds, dthdt

    @timeit
    def add_ops(self):
        '''
        ansatz adaptively expanding procedure in avqite by adding
        layer by layer.
        '''
        ntheta = len(self._params)
        npool = len(self._pool)
        if self._maxntheta > 0 and ntheta >= self._maxntheta:
            return

        icyc = 0
        # energy variance
        hvar = self._e2 - self._e**2

        # now layer
        inds_chk = list(range(npool))

        for _ in range(self._max_add):
            scores = self.get_scores(inds_chk)
            isort = scores.argsort()[::-1]
            ichoose = []
            nlayer = len(self._layer_range)
            qsupport = []
            if nlayer == 0:
                self._layer_range.append([0, 0])
            else:
                imin = self._layer_range[-1][1]
                self._layer_range.append([imin, imin])

            for i, idx in enumerate(isort):
                if scores[idx] < 0:
                    break

                if idx in ichoose or \
                    numpy.any([self._pool[idx][1][iq] != "I" for iq in qsupport]) or \
                    (nlayer > 0 and idx in self._ansatz_pids[self._layer_range[-1][0]: self._layer_range[-1][1]]):
                    continue

                ichoose.append(idx)
                inds_chk.remove(idx)

                self._ansatz_pids.append(idx)
                self._ansatz[0].append(self._pool[idx][0])
                self._ansatz[1].append(self._pool[idx][1])
                self.update_ngates()
                self._params.append(0)
                self._layer_range[-1][1] += 1

                for iq, s in enumerate(self._pool[idx][1]):
                    if s != "I":
                        qsupport.append(iq)
                if len(qsupport) == self._nq:
                    break

                if not self._tetras:
                    break

            # update mmat, vvec
            if len(ichoose) == 0:
                warnings.warn("no more unitaries can be added.")
                break

            diff, mmat, vvec, vecp_adds, dthdt = self.get_score(ichoose, mode=1)
            dist_p = diff + self._distp
            dist = hvar - dist_p
            self._distp = dist_p
            self._mmat = mmat
            self._vvec = vvec
            self._dthdt = dthdt
            self._vecp_list = numpy.vstack((self._vecp_list, vecp_adds))

            if self._mrank == 0:
                print("add op:", [self._ansatz[1][i] for i in range(self._layer_range[-1][0], self._layer_range[-1][1])])
                print("grad:", [f"{vvec[i]:.6f}" for i in range(self._layer_range[-1][0], self._layer_range[-1][1])])
                print(f"icyc = {icyc}, dist = {dist:.2e}, improving {diff:.2e}")
                assert(dist > -1e-12)
            icyc += 1

            if self._maxntheta > 0 and mmat.shape[0] >= self._maxntheta:
                break
            if dist < self._rcut:
                break
        if self._mrank == 0:
            print(f"number of layers in the ansatz: {len(self._layer_range)}")

    def get_mvhelp(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    @timeit
    def run(self, maxiter=-1):
        # change the representation of H if necessary.
        self._h = self._model._h.data.as_scipy()
        self.update_state()

        # keep time to save intermediate results
        if self._mrank == 0:
            twall0 = time.time()

        while True:
            self.one_step()
            var_h = self._e2 - self._e**2
            if self._mrank == 0:
                print(f'iter = {self._iter}, t = {self._t:.4f}, e = {self._e:.8f}, ' + \
                        f'vmax = {self._vmax:.2e} var_h = {var_h:.2e}')
                print(f'ngates: {self._ngates}', flush=True)
            if self._vmax < self._vtol or abs(self._tf - self._t) < 1e-7:
                break

            if self._mrank == 0:
                twall1 = time.time()
                if twall1 - twall0 > self._tsave:
                    self.psave_ansatz_simp()
                    twall0 = twall1

            self._iter += 1
            if maxiter > 0 and self._iter == maxiter:
                break

        if self._mrank == 0:
            print(f"total layers in ansatz: {len(self._layer_range)}")

    @timeit
    def one_step(self):
        self.set_par_states()
        amat = self.get_amat()
        cvec, e, e2 = self.get_cvec_phase()
        # McLachlan's principle, including global phase contribution
        m = amat.real
        # minus sign
        v = -cvec.real
        self._mmat = m
        self._vvec = v
        self._e, self._e2 = e, e2

        if len(v) > 0:
            dthdt = self.get_dthdt(m, v)
        else:
            dthdt = numpy.asarray([])
        # McLachlan distance
        dist_p = v.dot(dthdt)
        dist_p = dist_p.real
        dist_h2 = e2 - e**2
        dist = dist_h2 - dist_p

        self._distp = dist_p
        if len(dthdt) > 0:
            pthmax = numpy.max(numpy.abs(dthdt))
        else:
            pthmax = 0
        self._pthmax = pthmax
        self._dthdt = dthdt
        if self._mrank == 0:
            print(f"initial mcLachlan distance: {dist:.2e} pthmax: {pthmax:.2f}")
            assert(dist > -1e-12)
        if dist > self._rcut or len(self._params) == 0:
            self.add_ops()
        dt = self._dt
        if len(self._dthdt) > 0:
            pthmax = numpy.max(numpy.abs(self._dthdt))
            if self._mrank == 0:
                print(f"max element in dtheta/dt: {pthmax:.2f}")
            if pthmax > 0:
                # similar step size constraints for dtheta_max
                dt = min(self._dt, self._dt/pthmax)
        # up to self._tf if needed.
        dt = min(dt, self._tf-self._t)
        self._params = [p + pp*dt for p, pp in zip(self._params, self._dthdt)]
        self._t += dt
        # maximal gradient element
        self._vmax = numpy.max(numpy.abs(self._vvec))
        self.update_state()

    def get_dist(self):
        return self._e2 - self._e**2 - self._distp

    @timeit
    def set_par_states(self):
        ''' d |vec> / d theta.
        '''
        raise NotImplementedError

    @timeit
    def get_amat(self):
        np = len(self._params)
        if np == 0:
            amat = numpy.zeros((0, 0), dtype=numpy.complex128)
        else:
            amat = numpy.einsum("ik,jk->ij",
                    self._vecp_list.conj(),
                    self._vecp_list,
                    optimize=True,
                    )
        return amat

    @timeit
    def get_cvec_phase(self):
        np = len(self._params)
        # h |vec>
        hvec = self._h.dot(self._state)
        if np == 0:
            cvec = numpy.zeros(0, dtype=numpy.complex128)
        else:
            cvec = numpy.einsum("ij,j->i",
                    self._vecp_list.conj(),
                    hvec,
                    optimize=True,
                    )
        # energy
        e = numpy.vdot(self._state, hvec).real
        e2 = numpy.vdot(hvec, hvec).real

        return cvec, e, e2

    def save_ansatz(self):
        with h5py.File("ansatz.h5", "w") as f:
            # initial state params
            f["/params"] = self._params
            # ansatz operator labels
            f["/ansatz_code"] = self._ansatz[1]
            # ngates
            f["/ngates"] = self._ngates
            # reference state
            f["/ref_state"] = self._ref_state

    def psave_ansatz(self):
        with open("ansatz.pkle", "wb") as f:
            data = [self._ansatz,
                    self._params,
                    self._ref_state,
                    self._ngates,
                    self._layer_range,
                    self._t,
                    self._iter,
                    ]
            pickle.dump(data, f)

    def psave_ansatz_simp(self):
        # for continuous run
        with open("ansatz_simp.pkle", "wb") as f:
            data = [self._ansatz_pids,
                    self._params,
                    self._ngates,
                    self._layer_range,
                    self._t,
                    self._iter,
                    ]
            pickle.dump(data, f)

    def psave_ansatz_inp(self):
        # should be transferable.
        with open("ansatz_inp.pkle", "wb") as f:
            data = [self._ansatz[1],
                    self._params,
                    ]
            pickle.dump(data, f)

    def pload_ansatz(self):
        if self._mrank == 0:
            with open("ansatz.pkle", "rb") as f:
                data = pickle.load(f)
            data = MPI.pickle.dumps(data)
        else:
            data = None
        data = self._comm.bcast(data, root=0)
        [self._ansatz,
                self._params,
                self._ref_state,
                self._ngates,
                self._layer_range,
                self._t,
                self._iter,
                ] = MPI.pickle.loads(data)
        self._ansatz_pids = []
        labels = [p[1] for p in self._pool]
        for label in self._ansatz[1]:
            self._ansatz_pids.append(labels.index(label))

    def pload_ansatz_simp(self):
        if self._mrank == 0:
            with open("ansatz_simp.pkle", "rb") as f:
                data = pickle.load(f)
            data = MPI.pickle.dumps(data)
        else:
            data = None
        data = self._comm.bcast(data, root=0)

        [self._ansatz_pids,
                self._params,
                self._ngates,
                self._layer_range,
                self._t,
                self._iter,
                ] = MPI.pickle.loads(data)
        self.set_ansatz_from_pids()

    def set_ansatz_from_pids(self):
        for idx in self._ansatz_pids:
            self._ansatz[0].append(self._pool[idx][0])
            self._ansatz[1].append(self._pool[idx][1])

    def pload_ansatz_inp(self):
        if self._mrank == 0:
            with open("ansatz_inp.pkle", "rb") as f:
                data = pickle.load(f)
            data = MPI.pickle.dumps(data)
        else:
            data = None
        data = self._comm.bcast(data, root=0)
        [self._ansatz[1], self._params] = MPI.pickle.loads(data)

        self._ansatz_pids = []
        labels = [p[1] for p in self._pool]
        self._layer_range.append([0, 0])
        qsupport = []

        for i, label in enumerate(self._ansatz[1]):
            idx = labels.index(label)
            self._ansatz_pids.append(idx)
            self._ansatz[0].append(self._pool[idx][0])
            self.update_ngates(i)
            for iq, s in enumerate(label):
                if s != "I":
                    if iq in qsupport:
                        self._layer_range[-1][1] = i
                        self._layer_range.append([i, 0])
                        qsupport = []
                        for iqp, sp in enumerate(label[:iq]):
                            if sp != "I":
                                qsupport.append(iqp)
                    qsupport.append(iq)
        self._layer_range[-1][1] = len(self._ansatz[1])

    def save_state(self, t):
        with h5py.File("state.h5", "w") as f:
            f["t"] = t
            f["state"] = self._state


class ansatzSinglePool(ansatz):
    '''
    adaptvqite with single operator pool.
    '''
    @timeit
    def setup_pool(self, tol=1e-8):
        '''
        setup pool from incar.
        '''
        labels = self._model._incar["pool"]
        self._pool = [[self._model.label2opseq(s), s] for s in labels]
        if self._mrank == 0:
            print(f'pool dimension: {len(self._pool)}')

        # setup score bias
        inds = numpy.lexsort((labels, labels))
        self._scores_bias = inds*tol/len(labels)

    def update_ngates(self, idx=-1):
        '''
        update gate counts.
        '''
        label = self._ansatz[1][idx]
        iorder = len(label) - label.count('I')
        self._ngates[iorder-1] += 1

    def get_mvhelp(self,
            inds,  # indices of additonal unitaries to be appended
            ):
        ntheta = self._vvec.shape[0]
        ninds = len(inds)
        mmat = numpy.zeros((ntheta+ninds, ntheta+ninds))
        vvec = numpy.zeros(ntheta+ninds)
        if ntheta > 0:
            mmat[:ntheta, :ntheta] = self._mmat
            vvec[:ntheta] = self._vvec
        # H |vec>
        hvec = self._h.dot(self._state)
        vecp_adds = []
        for i in inds:
            vec = self._state
            for op in self._pool[i][0]:
                vec = op.dot(vec)
            vec *= -0.5j
            vecp_adds.append(vec)
        vecp_adds = numpy.asarray(vecp_adds)
        if ntheta > 0:
            mmat[ntheta:, :ntheta] = numpy.einsum("ik,jk->ij",
                    vecp_adds.conj(),
                    self._vecp_list,
                    optimize=True,
                    ).real
            mmat[:ntheta, ntheta:] = mmat[ntheta:, :ntheta].T
        mmat[ntheta:, ntheta:] = numpy.einsum("ik,jk->ij",
                vecp_adds.conj(),
                vecp_adds,
                optimize=True,
                ).real
        vvec[ntheta:] = -numpy.einsum("ik,k->i",
                vecp_adds.conj(),
                hvec,
                optimize=True,
                ).real
        return mmat, vvec, vecp_adds

    def get_state(self):
        return get_ansatz_state(self._params,
                        self._ansatz[0],
                        self._ref_state,
                        )

    @timeit
    def set_par_states(self, nblk=50):
        ''' d |vec> / d theta.
        '''
        np = len(self._params)
        vecp_list = numpy.zeros((np, 2**self._nq), dtype=numpy.complex128)

        for i in range(np):
            if i%self._msize == self._mrank:
                vec = self._ref_state
                for j, th, opseq in zip(itertools.count(), self._params, self._ansatz[0]):
                    # factor of 0.5 difference from the main text.
                    vec_cos = numpy.cos(0.5*th)*vec
                    for op in opseq:
                        vec = op.dot(vec)
                    vec *= -1j*numpy.sin(0.5*th)
                    vec += vec_cos
                    if j == i:
                        for op in opseq:
                            vec = op.dot(vec)
                        vec *= -0.5j
                    vecp_list[i, :] = vec
        for i in range(np//nblk+1):
            ista = i*nblk
            iend = min(np, ista+nblk)
            if iend > ista:
                vecp_list[ista:iend, :] = self._comm.allreduce(vecp_list[ista:iend, :])
        self._vecp_list = vecp_list


def get_ansatz_state(params, ansatz, ref_state):
    state = ref_state
    for theta, opseq in zip(params, ansatz):
        # use self-inverse of Pauli string
        state_cos = numpy.cos(0.5*theta)*state
        for op in opseq:
            state = op.dot(state)
        state *= -1j*numpy.sin(0.5*theta)
        state += state_cos
    return state
