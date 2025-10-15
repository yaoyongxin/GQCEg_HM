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
from qiskit.circuit import Parameter, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.primitives import StatevectorEstimator
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
        depolarizing_error,
        pauli_error,
        )

def add_pauli_rotation_gate(
    qc: "QuantumCircuit",
    pauli_string: str,
    theta: float,
    decompose_rzz: bool = True
):
    """
    Appends a Pauli rotation gate to a QuantumCircuit.
    Convention for Pauli string ordering is opposite to the Qiskit convention.
    For example, in string "XYZ" Pauli "X" acts on the first qubit.

    Parameters
    ----------
    qc : "QuantumCircuit"
        Qiskit "QuantumCircuit" to which the Pauli rotation gate is appended.
    pauli_string : str
        Pauli string defining the rotation.
    theta : float
        Rotation angle.
    decompose_rzz : bool
        If decompose_rzz==True, all rzz gates are decompsed into cx-rz-cx.
        Otherwise, the final circuit contains rzz gates.

    Returns
    -------
    ansatz_adaptvqite : List[str]
        List of Pauli strings entering the ansatz.
    params_adaptvqite : List[float64]
        Parameters (angles) of the ansatz.
    """

    #if qc.num_qubits != len(pauli_string):
    #    raise ValueError("Circuit and Pauli string are of different size")
    if all([pauli=='I' or pauli=='X' or pauli=='Y' or pauli=='Z'
            for pauli in pauli_string])==False:
        raise ValueError("Pauli string does not have a correct format")

    nontriv_pauli_list = [(i,pauli)
                        for i,pauli in enumerate(pauli_string) if pauli!='I']
    for (i,pauli) in nontriv_pauli_list:
        if pauli=='X':
            qc.h(i)
        if pauli=='Y':
            qc.sdg(i)
            qc.h(i)
    for list_ind in range(len(nontriv_pauli_list)-2):
        qc.cx(nontriv_pauli_list[list_ind][0],nontriv_pauli_list[list_ind+1][0])
    if decompose_rzz==True:
        qc.cx(
            nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
            nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
        qc.rz(theta,nontriv_pauli_list[len(nontriv_pauli_list)-1][0])
        qc.cx(
            nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
            nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
    if decompose_rzz==False:
        qc.rzz(
            theta,
            nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
            nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
    for list_ind in reversed(range(len(nontriv_pauli_list)-2)):
        qc.cx(nontriv_pauli_list[list_ind][0],nontriv_pauli_list[list_ind+1][0])
    for (i,pauli) in nontriv_pauli_list:
        if pauli=='X':
            qc.h(i)
        if pauli=='Y':
            qc.h(i)
            qc.s(i)
    return qc

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
        self._init_qc = QuantumCircuit(self._num_qubits + 4)
        self._init_qc.h(0)
        for i in range(self._num_qubits+1):
            self._init_qc.cx(i,i+1)
        #If the reference state contains "1"s, adds corresponding bit-flips.
        if self._ref_state.count('1')%2 != 0:
            self._init_qc.x(0)
        if all([(el=='0') or (el=='1') for el in self._ref_state]):
            self._init_qc.x(
                    [i+1 for i,el in enumerate(self._ref_state) if el=='1']
                    )
        else:
            raise ValueError(
                    "Reference state is supposed to be a string of 0s and 1s"
                    )

        #Constructs a final QuantumCircuit object that represents the ansatz


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


    def pauli_rotation_gate(
        self,
        theta: float,
        pauli_string: str,
    ):
        """
        Generates a Pauli string rotation gate.

        Parameters
        ----------
        theta : float
            Pauli rotation angle.

        Returns
        -------
        gate : Qiskit instruction
        """
        operator = SparsePauliOp(pauli_string)
        gate = PauliEvolutionGate(operator, time = theta/2)
        return gate


    def circuit_construction(
        self,
        num_bar=1,
        decompose_rzz: bool=False,
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
        new_str=[]
        for st in self._ansatz_adaptvqite:
            new_str.append('I'+st+'Z')
        len_bar=len(new_str)//num_bar
        bar_count=0
        for i, pauli_string in enumerate(new_str):
            if (i%len_bar==0) and (i!=0) and (bar_count != num_bar-1):
                qc.barrier()
                bar_count+=1
            theta = self._params_ansatz[i]
            qc = add_pauli_rotation_gate(
                qc,
                pauli_string,
                theta,
                decompose_rzz
                )
        qc.barrier()
        return qc


def add_meas(qc,num):
    qc_Size=qc.qregs[0]._size
    qr=QuantumRegister(qc_Size, 'q')
    cr=ClassicalRegister(2*(num)+(qc_Size-2), 'c')
    qc_new = QuantumCircuit(qr,cr)

    syn_count=0
    for CirIns in qc:
        op=CirIns.operation
        if op.name=='barrier':
            qc_new.h(qr[-1])
            qc_new.cx(qr[-1],qr[0])
            qc_new.cx(qr[0],qr[-2])
            qc_new.cx(qr[1],qr[-2])
            qc_new.cx(qr[-1],qr[1])

            nbb=((qc_Size-4)//2-1)
            for i in range(nbb):
                qc_new.cx(qr[-1],qr[2+2*i])
                qc_new.cx(qr[2+2*i],qr[-2])
                qc_new.cx(qr[-1],qr[3+2*i])
                qc_new.cx(qr[3+2*i],qr[-2])

            qc_new.cx(qr[-1],qr[-4])
            qc_new.cx(qr[-4],qr[-2])
            qc_new.cx(qr[-3],qr[-2])
            qc_new.cx(qr[-1],qr[-3])

            qc_new.h(qr[-1])

            qc_new.measure(qr[-1],cr[2*syn_count])
            qc_new.measure(qr[-2],cr[2*syn_count+1])
            #qc_new.barrier()
            syn_count+=1
        else:
            qc_new.append(CirIns)
    return qc_new

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

N_mea=1

qc = model.circuit_construction(num_bar=N_mea)

#qc = add_meas(qc,N_mea)

# unfinished.

# print(qc)
print(f"original circuit depth: {qc.depth()}")
gates = [ins.name for ins in qc]
print("original unique gates:", set(gates), "cx:", gates.count("cx"), "rzz:", gates.count("rzz"))
print("original non-local gates:", qc.num_nonlocal_gates())
basis_gates = ['x','z', 'rx', 'rzz', 'rz', 'cx']
qc = transpile(qc, basis_gates=basis_gates,optimization_level=3)
# print(qc)
print(f"circuit depth: {qc.depth()}")
# unique gates
gates = [ins.name for ins in qc]
print("unique gates:", set(gates), "cx:", gates.count("cx"), "rzz:", gates.count("rzz"))
print("non-local gates:", qc.num_nonlocal_gates())

