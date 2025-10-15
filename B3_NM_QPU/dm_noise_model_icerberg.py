"""
This notebook demonstrates how to construct a Qiskit QuantumCircuit from an
AVQITE file, and use it to compute the expectation values of some observables
using Qiskit StatevectorEstimator.

Packages information:
---------------------
python version = 3.11 (python version >= 3.7 should suffice)
qiskit version > 1
qiskit[visualization] is required for visualization functionality
"""


import numpy as np
import pickle
from typing import (
    List,
    Optional,
    Tuple,
    Union
)
from qiskit.circuit import Parameter, QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.primitives import StatevectorEstimator
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
        depolarizing_error,
        pauli_error,
        )



class QiskitCircuitGeneratorAVQITE:
    """
    Class for reading out the AVQITE ansatz and incar files, and constructing
    a Qiskit QuantumCircuit object representing the ansatz.

    Attributes
    ----------
    ansatz_filename : str
        File name of the ansatz file, including the path.
        For this implementation, "ansatz_inp.pkle" type of ansatz files is used.
    incar_filename : str
        File name of the incar file, including the path.
        Incar file is used to read out the reference state.
    """
    def __init__(
        self,
        ansatz_filename: str,
        incar_filename: str
    ):
        self._ansatz_filename = ansatz_filename
        self._incar_filename = incar_filename

        #Reads out the operator list and the parameter list of the ansatz.
        (self._ansatz_adaptvqite,
         self._params_ansatz) = self.read_adaptvqite_ansatz(ansatz_filename)

        #Reads out the number of qubits.
        self._num_qubits = len(self._ansatz_adaptvqite[0])

        #Reads out the incar file.
        with open(incar_filename) as fp:
            incar_content = fp.read()
        ref_st_r_pos = incar_content.find("ref_state")
        #Reads out the reference state from the incar file.
        self._ref_state = incar_content[
                            ref_st_r_pos+13:ref_st_r_pos+13+self._num_qubits
                            ]

        #Initializes a QuantumCircuit object.
        self._init_qc = QuantumCircuit(self._num_qubits)

        #If the reference state contains "1"s, adds corresponding bit-flips.
        if all([(el=='0') or (el=='1') for el in self._ref_state]):
            self._init_qc.x(
                [i for i,el in enumerate(self._ref_state) if el=='1']
            )
        else:
            raise ImportError(
                "Reference state is supposed to be a string of 0s and 1s"
            )
        self.circuit_construction()

    def read_adaptvqite_ansatz(
        self,
        filename: str
    ):
        """
        Reads the ansatz from a file resulting from adaptvqite calculation.

        Parameters
        ----------
        filename : str
            Name of a file containing the results of adaptvqite calculation.
            Has to be given in .pkle format.

        Returns
        -------
        ansatz_adaptvqite : List[str]
            List of Pauli strings entering the ansatz.
        params_adaptvqite : List[float64]
            Parameters (angles) of the ansatz.
        """
        if filename[-5:] != '.pkle':
            raise ImportError("Ansatz file should be given in .pkle format")
        with open(filename, 'rb') as inp:
            data_inp = pickle.load(inp)
            ansatz_adaptvqite = data_inp[0]
            params_adaptvqite = data_inp[1]
        return ansatz_adaptvqite, params_adaptvqite

    def circuit_construction(
        self,
        decompose_rzz: bool = False,
        ):
        """
        Constructs a "QuantumCircuit" representing the AVQITE ansatz.

        Parameters
        ----------
        decompose_rzz : bool
            If decompose_rzz==True, all rzz gates are decompsed into cx-rz-cx.
            Otherwise, the final circuit contains rzz gates.

        Returns
        -------
        qc : "QuantumCircuit"
            "QuantumCircuit" representing the AVQITE ansatz.
        """
        qc = self._init_qc.copy()

        #Adds (potentially multi-qubit) Pauli rotation gates to the circuit
        #from the ansatz using "add_pauli_rotation_gate" function.
        #Option "decompose_rzz=False" can be used to not decompose rzz gates
        #into cx-rz-cx
        for i, pauli_string in enumerate(self._ansatz_adaptvqite):
            theta = self._params_ansatz[i]
            qc = add_pauli_rotation_gate(
                qc,
                pauli_string,
                theta,
                decompose_rzz,
            )
        self.qc = qc


def get_qc_noise(qc_inp, p_init, p1, p2):
    # no measurement yet
    Noise_Model={
            "init_error": pauli_error([('X', p_init), ('I', 1 - p_init)]),
            "Two_q_err": depolarizing_error(p2, 2),
            "One_q_err": pauli_error([('X', p1/3), ('Y', p1/3), ('Z', p1/3), ('I', 1 - p1)]),
            }
    qc = QuantumCircuit(qc_inp.qregs[0])

    # init error
    for q in qc.qregs[0]:
        qc.append(Noise_Model["init_error"],[q])

    for ins in qc_inp:
        assert(ins.operation.name != "measure"), "no measure expected."
        qc.append(ins)
        if ins.operation.num_qubits == 1:
            qc.append(Noise_Model["One_q_err"], ins.qubits)
        elif ins.operation.num_qubits == 2:
            qc.append(Noise_Model["Two_q_err"], ins.qubits)
        elif ins.operation.name == "barrier":
            pass
        else:
            raise ValueError("Invalid instruction.")
    return qc


def get_norm_all(qc_inp, qc_noise, p_meas, p1, p2, shots=1e5):
    # no measurement yet
    Noise_Model={
            "meas_error": pauli_error([('X', p_meas), ('I', 1 - p_meas)]),
            "Two_q_err": depolarizing_error(p2, 2),
            "One_q_err": pauli_error([('X', p1/3), ('Y', p1/3), ('Z', p1/3), ('I', 1 - p1)]),
            }
    qc = qc_noise.copy()

    for ins in qc_inp[::-1]:
        assert(ins.operation.name != "measure"), "no measure expected."
        if ins.operation.name == "barrier":
            qc.append(ins)
        else:
            qc.append(ins.operation.inverse(), ins.qubits)
        if ins.operation.num_qubits == 1:
            qc.append(Noise_Model["One_q_err"], ins.qubits)
        elif ins.operation.num_qubits == 2:
            qc.append(Noise_Model["Two_q_err"], ins.qubits)
        elif ins.operation.name == "barrier":
            pass
        else:
            raise ValueError("Invalid instruction.")

    # add measure all in z basis
    for q in qc.qregs[0]:
        qc.append(Noise_Model["meas_error"], [q])
    qc.measure_all()

    simulator = AerSimulator(method='density_matrix')
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    count_zeros = counts['0'*qc.num_qubits]
    norm = count_zeros/shots
    print(f"norm: {norm}")
    return norm


def get_exp_val(obs, qc_noise, p1, p_meas, shots=1e5):
    Noise_Model={
            "One_q_err": pauli_error([('X', p1/3), ('Y', p1/3), ('Z', p1/3), ('I', 1 - p1)]),
            "meas_error": pauli_error([('X',p_meas), ('I', 1 - p_meas)]),
            }
    simulator = AerSimulator(method='density_matrix')
    res = 0
    for op in obs[0]:
        circ = qc_noise.copy()
        plable = op.paulis[0].to_label()

        if plable.count("I") == len(plable):
            res += op.coeffs[0].real
            continue

        for i, s in enumerate(plable[::-1]):
            if s == "Y":
                circ.sdg(circ.qregs[0][i])
                circ.append(Noise_Model["One_q_err"], [circ.qregs[0][i]])
                circ.h(circ.qregs[0][i])
                circ.append(Noise_Model["One_q_err"], [circ.qregs[0][i]])
            elif s == "X":
                circ.h(circ.qregs[0][i])
                circ.append(Noise_Model["One_q_err"], [circ.qregs[0][i]])

        ncregs = len(plable) - plable.count('I')
        circ.add_register(ClassicalRegister(ncregs, name='c'))
        icreg = 0
        for i, q in enumerate(circ.qregs[0]):
            if plable[::-1][i] != "I":
                circ.append(Noise_Model["meas_error"], [q])
                circ.measure(q, circ.cregs[0][icreg])
                icreg += 1

        # print(plable, "\n", circ)

        result = simulator.run(circ, shots=shots).result()
        val = 0
        for key, count in result.get_counts().items():
            iflag = np.sum(np.array(list(key), dtype=int))%2
            val = val + (-1)**iflag*count/shots
        res += val*op.coeffs[0].real

    return res


#qc is the circuit in the final from that we need
f_incar = "../incar"
model = QiskitCircuitGeneratorAVQITE(
        ansatz_filename='ansatz_inp.pkle',
        incar_filename=f_incar,
        )
qc = model.qc
#print(qc)
print(f"circuit depth: {qc.depth()}")


#Computing expectation values of observables
import json
data_incar = json.load(open(f_incar, "r"))
obs_list_2D = data_incar["observables"]

# Converts the list of observables into the Qiskit format

obs_list_qiskit=[]
const_list = np.zeros(len(obs_list_2D))

for i, obs_row in enumerate(obs_list_2D):
    pauli_strings_list = []
    params_list = []
    for obs in obs_row:
        pauli = "".join([i for i in obs if (i=="I" or i=="X" or i=="Y" or i=="Z")])
        coeff = float("".join([i for i in obs if (i=="-" or i=="." or i.isdigit())]))

        # remove trivial identity
        if pauli != "I"*len(pauli):
            pauli_strings_list.append(pauli[::-1])
            params_list.append(coeff)
        else:
            const_list[i] = coeff


    obs_list_qiskit.append([SparsePauliOp(pauli_strings_list, params_list)])

# Computes expectation values using Qiskit StatevectorEstimator
estimator = StatevectorEstimator()
pub = (qc, obs_list_qiskit)
job = estimator.run([pub])
result = job.result()[0]

print("exact reference results:")
for ob, val, vadd in zip(obs_list_qiskit, result.data.evs, const_list):
    #print(f"{ob}: {val+vadd}, {val}")
    print(f"{val[0]+vadd:10.6f}, {val[0]:10.6f}")

# unique gates
gates = [ins.name for ins in qc]
print("unique gates:", set(gates), "cx:", gates.count("cx"))
print("non-local gatyes:", qc.num_nonlocal_gates())

# noise model

scale = 0.1
p_init = p1 = 4e-4*scale
p_meas = p2 = 3e-3*scale

# trasnpile?
# simulator = AerSimulator(method='automatic')
# qc = transpile(qc, simulator, optimization_level=3)
# gates = set([ins.name for ins in qc])
# print("unique gates in transpiled circ:", gates)
# print("non-local gates in transpiled circ:", qc.num_nonlocal_gates())

qc_noise = get_qc_noise(qc, p_init, p1, p2)

# print(qc)
# print(qc_noise)

vals = []
for obs in obs_list_qiskit:
    res = get_exp_val(obs, qc_noise, p1, p_meas)
    vals.append(res)

for ob, val, vadd, valp in zip(obs_list_qiskit, result.data.evs, const_list, vals):
    print(f"{valp:10.6f} vs {val[0]+vadd:10.6f}, {val[0]:10.6f}: {valp/val[0]:10.6f}")


get_norm_all(qc, qc_noise, p_meas, p1, p2, shots=1e5)
