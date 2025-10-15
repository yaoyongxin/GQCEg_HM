# author: Yongxin Yao (yxphysice@gmail.com)
from mpi4py import MPI
import numpy, json
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, Qobj
from timing import timeit


class model:
    '''qubit model defined in incar file.
    '''
    def __init__(self,
            wpenalty=10.,  # prefactor of pentalty terms
            hmode=0,       # h in mo representation.
            ):
        comm = MPI.COMM_WORLD
        self._mrank = comm.Get_rank()

        self._wpenalty = wpenalty
        self._hmode = hmode
        self.load_incar()
        self.set_h()
        self.add_npenalty()

    def get_h_expval(self, vec):
        '''
        get Hamiltonian expectation value.
        '''
        # convert the state vector vec to Q object if it is an array.
        if isinstance(vec, numpy.ndarray):
            vec = Qobj(vec)
        # return the real expectation value.
        return self._h.matrix_element(vec, vec).real

    def get_loweste_states(self, eigvals=7, sparse=True):
        '''
        get the lowest three eigenvalues and eigenstates.
        '''
        # get the lowest three eigenvalues and eigenstates.
        w, v = self._h.eigenstates(eigvals=eigvals,
                sparse=sparse and self._nsite>4,
                )
        return w, v

    def load_incar(self):
        self._incar = json.load(open("incar", "r"))

    def set_sops(self):
        '''
        set up site-wise sx, sy, sz operators.
        '''
        self._ops = {}
        self._ops["X"], self._ops["Y"], self._ops["Z"] = \
                get_sxyz_ops(self._nsite)

    @timeit
    def set_h(self):
        if self._hmode == 0:
            hs_list = self._incar["h"]
        else:
            hs_list = self._incar["hao"]
        if self._mrank == 0:
            print(f'Hamiltonian terms: {len(hs_list)}', flush=True)
        self._nsite = len(hs_list[0].split("*")[1])
        self.set_sops()
        self._h = self.coeflabels2op(hs_list)

    @timeit
    def add_npenalty(self):
        '''add (ntot_op -n_e)**2 if ntot.inp is present.
        '''
        op_pentaly = self.get_npenalty()
        if op_pentaly is not None:
            self._h += op_pentaly

    def get_npenalty(self):
        if "ntot" in self._incar:
            ns_list = self._incar["ntot"]
            n_op = self.coeflabels2op(ns_list)
            nume = self._incar["nume"]
            op_pentaly = self._wpenalty*(n_op - nume)**2
            return op_pentaly
        else:
            return None

    def coeflabels2op(self, clabels):
        op = 0
        for clabel in clabels:
            coef, label = clabel.split("*")
            op1 = self.label2op(label)
            op += op1*float(coef)
        return op

    def coeflabels2coefops(self, clabels, skipone=True):
        cop = []
        one = 'I'*self._nsite
        for clabel in clabels:
            coef, label = clabel.split("*")
            if label == one and skipone:
                continue
            op = self.label2op(label)
            cop.append([float(coef), op])
        return cop

    def label2op(self, label):
        ops = {"I": qeye(2),
                "X": sigmax(),
                "Y": sigmay(),
                "Z": sigmaz(),
                }
        op_list = [ops[s] for s in label]
        return tensor(op_list)

    def label2opseq(self, label):
        op = []
        for i, s in enumerate(label):
            if s in ["X", "Y", "Z"]:
                op.append(self._ops[s][i])
        return op


def get_sxyz_ops(nsite):
    '''
    set up site-wise sx, sy, sz operators.
    '''
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sx_list = []
    sy_list = []
    sz_list = []

    op_list = [si for i in range(nsite)]
    for i in range(nsite):
        op_list[i] = sx
        sx_list.append(tensor(op_list).data.as_scipy())
        op_list[i] = sy
        sy_list.append(tensor(op_list).data.as_scipy())
        op_list[i] = sz
        sz_list.append(tensor(op_list).data.as_scipy())
        # reset
        op_list[i] = si
    return [sx_list, sy_list, sz_list]
