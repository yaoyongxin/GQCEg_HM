import h5py, numpy, json, itertools
from openfermion import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner
from openfermion.hamiltonians.special_operators import (
        s_squared_operator,
        number_operator,
        )
from openfermion.utils import hermitian_conjugated


def gen_incar(fname, path, tol=1e-12, weight=10.):
    # spin-index faster
    with h5py.File(fname, "r") as f:
        D = f[f"{path}/D"][()]
        H1E = f[f"{path}/H1E"][()]
        LC = f[f"{path}/LC"][()]
        V2E = f[f"{path}/V2E"][()]

    norbs_spatial_tot = sum(D.shape)//2
    norbs_phy = D.shape[1]

    fop = FermionOperator()
    # H1E
    for i, j in numpy.argwhere(abs(H1E)>tol):
        i, j = [int(x) for x in [i, j]]
        fop += FermionOperator(((i,1),(j,0)),H1E[i,j].real)
    # V2E
    for i, j, k, l in numpy.argwhere(abs(V2E)>tol):
        i, j, k, l = [int(x) for x in [i, j, k, l]]
        fop += FermionOperator(((i,1),(k,1),(l,0),(j,0)),0.5*V2E[i,j,k,l].real)
    # LC
    for a, b in numpy.argwhere(abs(LC)>tol):
        a, b = [int(x) for x in [a, b]]
        fop += FermionOperator(((b+norbs_phy,0),(a+norbs_phy,1)),LC[a,b].real)
    # D
    for a, α in numpy.argwhere(abs(D)>tol):
        a, α = [int(x) for x in [a, α]]
        dop = FermionOperator(((α,1),(a+norbs_phy,0)),D[a,α].real)
        fop += dop + hermitian_conjugated(dop)

    # S^2 penalty
    sop = s_squared_operator(norbs_spatial_tot)
    fop += weight*(sop)
    # N_tot penalty for half-filling
    nop = number_operator(norbs_spatial_tot*2, parity=-1)
    # seems not necessary
    fop += weight*(nop-norbs_spatial_tot)**2

    # exact checking
    check_gs(fop, sop, nop, norbs_spatial_tot)

    qop = jordan_wigner(fop)
    qop.compress(abs_tol=tol)
    data = {}
    data["h"] = qubit_operator_to_strings(qop, norbs_spatial_tot*2)
    print(f"Hamiltonian terms: {len(data['h'])}")
    data["pool"] = get_xypool(norbs_spatial_tot*2)
    # data["pool"] = get_xypool_local(norbs_spatial_tot*2)
    print(f"pool size: {len(data['pool'])}")
    data["nume"] = norbs_spatial_tot
    data["ref_state"] = "1"*norbs_spatial_tot+"0"*norbs_spatial_tot

    # denosty matrix, spin up component, spin-faster indices.
    ob_ops = []
    for i in range(norbs_spatial_tot):
        for j in range(i, norbs_spatial_tot):
            ob_ops.append(FermionOperator(((i*2,1),(j*2,0)),1.))
    # double occupancy
    ob_ops.append(FermionOperator(((0,1),(0,0),(1,1),(1,0)),1.))
    ob_ops = [qubit_operator_to_strings(jordan_wigner(op), norbs_spatial_tot*2)
            for op in ob_ops]
    data["observables"] = ob_ops

    with open('incar', 'w') as f:
        json.dump(data, f, indent=4)


def check_gs(op, sop, nop, nocc):
    # check grounmd state energy
    from openfermion.linalg import get_sparse_operator
    hmat = get_sparse_operator(op).todense()
    from numpy.linalg import eigh
    w, v = eigh(hmat)
    print(f"Exact gs energy: {w[0]:.6f}, followed by: {w[1:9]} ")
    smat = get_sparse_operator(sop)
    s2avg = v[:,1].T.conj().dot(smat.dot(v[:,1]))[0,0]
    print(f"s2avg: {s2avg:.6f}")
    assert(abs(s2avg) < 1e-6)
    nmat = get_sparse_operator(nop)
    navg = v[:,1].T.conj().dot(nmat.dot(v[:,1]))[0,0]
    print(f"navg: {navg:.6f}")
    assert(abs(navg - nocc) < 1e-6)


def qubit_operator_to_strings(qubit_op, num_qubits, tr_symmetry=True):
    terms_list = []
    for term, coefficient in qubit_op.terms.items():
        # Because of time-reversal symmetry, the pauli terms with complex coeffcient will not contribute.
        if tr_symmetry:
            if abs(coefficient.imag) > 1e-6:
                continue
            coefficient = coefficient.real

        # Initialize a list representing the identity operator on each qubit
        pauli_string = ['I'] * num_qubits
        for index, pauli in term:
            pauli_string[index] = pauli

        # Convert list to string and format the coefficient
        pauli_string = ''.join(pauli_string)
        term_string = f"{coefficient}*{pauli_string}"
        terms_list.append(term_string)

    return terms_list


def get_xypool(nq):
    # Generate combinations with exactly 2 and 4 letters in [X, Y]
    combinations_2 = generate_xy_combinations(nq, 2)
    combinations_4 = generate_xy_combinations(nq, 4)

    # Combine the lists and filter to ensure an odd number of Y's
    all_combinations = combinations_2 + combinations_4

    return all_combinations


def generate_xy_combinations(nq, num_letters):
    indices_combinations = itertools.combinations(range(nq), num_letters)
    xy_combinations = []
    for indices in indices_combinations:
        for letter_permutation in itertools.product('XY', repeat=num_letters):
            if letter_permutation.count("Y")%2 == 0:
                continue
            combo = ['I'] * nq
            for index, letter in zip(indices, letter_permutation):
                combo[index] = letter
            xy_combinations.append(''.join(combo))
    return xy_combinations


# not working usually.
def get_xypool_local(nq):
    pool = []
    for pattern in ["XY", "YX", "XYYY", "YXYY", "YYXY", "YYYX"]:
        for i in range(nq-len(pattern)+1):
            pool.append("I"*i + pattern + "I"*(nq-len(pattern)-i))

    return pool



if __name__ == "__main__":
    gen_incar(fname="hembed_cyc2_list.h5", path="/iter_13/u_2.50/")
