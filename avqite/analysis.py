#!/usr/bin/env python
from mpi4py import MPI
from model import model
from ansatz import ansatzSinglePool
import argparse,sys
import numpy


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--hmode", type=int, default=0,
        help="hmode. 0: h in mo; 1: h in ao. dflt: 0.")
parser.add_argument("-c", "--rcut", type=float, default=1.e-2,
        help="McLachlan distance cut-off. dflt: 0.01")
parser.add_argument("-f", "--fcut", type=float, default=1.e-2,
        help="invidual unitary cut-off ratio. dflt: 0.01")
parser.add_argument("-m", "--maxadd", type=int, default=5,
        help="Mix. allowed unitaries to be added at one iteration. dflt: 5.")
parser.add_argument("-b", "--bound", type=float, default=10,
        help="Bounds for dtheta/dt: [-b, b]. dflt: 10")
parser.add_argument("--delta", type=float, default=1e-4,
        help="Tikhonov parameter. dflt: 1e-4. Nagative value switch on lsq.")
parser.add_argument("-t", "--dt", type=float, default=0.02,
        help="Time step size. dflt: 0.02")
args = parser.parse_args()

mdl = model(hmode=args.hmode)
ans = ansatzSinglePool(mdl,
        rcut=args.rcut,
        fcut=args.fcut,
        max_add=args.maxadd,
        bounds=[-args.bound, args.bound],
        dt=args.dt,
        delta=args.delta,
        )

comm = MPI.COMM_WORLD
m_rank = comm.Get_rank()

print(f"final number of ansatz parameters: {len(ans._params)}")

w, v = mdl.get_loweste_states()
if m_rank == 0:
    print("lowest energies: ", w)
    print('reference state energy:', mdl.get_h_expval(ans._ref_state))
    print('ansatz state energy:', mdl.get_h_expval(ans.get_state()))

if False:
    # check domain walls
    op_zz = mdl.coeflabels2op(mdl._incar["zz"])
    op_zz = op_zz.data.as_scipy()

    nbond = 4
    v0 = numpy.array(v[0].data.todense()).reshape(-1)
    print((nbond-v0.conj().dot(op_zz.dot(v0)))/2)
    v0 = numpy.array(v[1].data.todense()).reshape(-1)
    print((nbond-v0.conj().dot(op_zz.dot(v0)))/2)
    v0 = numpy.array(v[2].data.todense()).reshape(-1)
    print((nbond-v0.conj().dot(op_zz.dot(v0)))/2)

    v0 = ans._ref_state
    print((nbond-v0.conj().dot(op_zz.dot(v0)))/2)

if True:
    # one particle density matrix in one spin channel
    dm = numpy.zeros((mdl._nsite//2, mdl._nsite//2))
    dm_ref = numpy.zeros((mdl._nsite//2, mdl._nsite//2))
    ij = 0
    v0 = ans.get_state()
    v0_ref = v[0][:, 0]
    print(f"fidelity: {abs(v0.dot(v0_ref))**2}")
    for i in range(mdl._nsite//2):
        for j in range(i, mdl._nsite//2):
            op = mdl.coeflabels2op(mdl._incar["observables"][ij])
            op = op.data.as_scipy()
            dm[i, j] = dm[j, i] = v0.conj().T.dot(op.dot(v0)).real
            dm_ref[i, j] = dm_ref[j, i] = v0_ref.conj().dot(op.dot(v0_ref)).real
            ij += 1

            print((mdl._incar["observables"][ij-1]))
            print("exp_val:", dm[i, j])

    print(dm)
    print("vs ref:")
    print(dm_ref)
    print(f"dm max diff: {abs(dm - dm_ref).max()}")

    import h5py
    with h5py.File("dm.h5", "w") as f:
        f["dm"] = dm
        f["dm_ref"] = dm_ref

    # double occupancy
    op = mdl.coeflabels2op(mdl._incar["observables"][ij])
    op = op.data.as_scipy()
    docc = v0.conj().dot(op.dot(v0))
    print("docc:", docc)
    docc = v0_ref.conj().dot(op.dot(v0_ref))
    print("docc_ref:", docc)
