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
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.primitives import StatevectorEstimator



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

        #Constructs a QuantumCircuit object with high-level Pauli evolution
        #gates.
        #This circuit is not decomposed into primitive gates and does not
        #include the initial bit-flips of the reference state
        qc_interm = self.circuit_construction(
                                    self._ansatz_adaptvqite,
                                    self._params_ansatz
                                    )
        #Constructs a final QuantumCircuit object that represents the ansatz
        self.qc = self._init_qc.compose(qc_interm.decompose())


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
        operator_list,
        parameter_list
    ):
        """
        Constructs a QuantumCircuit object consisting of high-level Pauli
        evolution gates from the ansatz operator and parameter list.
        Qiskit convention, by which the rightmost operator in a Pauli string
        acts on the qubit0, is taken into account by reversing the order of
        each Pauli string.

        Returns
        -------
        qc : QuantumCircuit
        """
        qc = QuantumCircuit(self._num_qubits)
        for i, pauli_string in enumerate(operator_list):
            theta = parameter_list[i]
            qc.append(
                self.pauli_rotation_gate(theta, pauli_string[::-1]),
                range(self._num_qubits)
                )
        return qc


#qc is the circuit in the final from that we need
model = QiskitCircuitGeneratorAVQITE(ansatz_filename='ansatz_inp.pkle', incar_filename='incar')
qc = model.qc
#print(qc)
print(f"circuit depth: {qc.depth()}")


#Computing expectation values of observables
obs_list_2D = [
        [
            "0.5*IIIIIIII",
            "-0.5*ZIIIIIII"
        ],
        [
            "0.25*YZYIIIII",
            "0.25*XZXIIIII"
        ],
        [
            "0.25*YZZZYIII",
            "0.25*XZZZXIII"
        ],
        [
            "0.25*YZZZZZYI",
            "0.25*XZZZZZXI"
        ],
        [
            "0.5*IIIIIIII",
            "-0.5*IIZIIIII"
        ],
        [
            "0.25*IIYZYIII",
            "0.25*IIXZXIII"
        ],
        [
            "0.25*IIYZZZYI",
            "0.25*IIXZZZXI"
        ],
        [
            "0.5*IIIIIIII",
            "-0.5*IIIIZIII"
        ],
        [
            "0.25*IIIIYZYI",
            "0.25*IIIIXZXI"
        ],
        [
            "0.5*IIIIIIII",
            "-0.5*IIIIIIZI"
        ],
        [
            "0.25*IIIIIIII",
            "-0.25*IZIIIIII",
            "-0.25*ZIIIIIII",
            "0.25*ZZIIIIII"
        ]
    ]


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

for ob, val, vadd in zip(obs_list_qiskit, result.data.evs, const_list):
    print(f"{ob}: {val+vadd}, {val}")
    # print(f"{val[0]+vadd:10.6f}, {val[0]:10.6f}")


if False:
    # IBM Circuit function
    from qiskit_ibm_catalog import QiskitFunctionsCatalog
    from qiskit_ibm_runtime import QiskitRuntimeService

    catalog = QiskitFunctionsCatalog()
    function = catalog.load("ibm/circuit-function")
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False)
    print(f"backend: {backend}")

    job = function.run(
            backend_name=backend.name,
            pubs=[pub],
            options = {"mitigation_level": 3, "default_precision": 0.01},
            )

    print(job)
    print(job.status())

if True:
    from qiskit_ionq import IonQProvider
    from qiskit import transpile
    provider = IonQProvider()
    simulator_backend = provider.get_backend("ionq_simulator", gateset="native")
    qc_native = transpile(qc, backend=simulator_backend)
    print(qc_native)
