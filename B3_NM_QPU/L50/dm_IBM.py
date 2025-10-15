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
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager



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

        # to debug
        # self.qc = self._init_qc


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


# pytket QubitPauliOperator
from pytket.utils.operators import QubitPauliOperator
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit

def pauliStringsToQubitPauliOperator(pauli_strings, coeffs):
    ixyz = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
    assert(len(pauli_strings) == len(coeffs))
    term_sum = {}
    for string, coef in zip(pauli_strings, coeffs):
        tmp = {}
        # convention
        for i, s in enumerate(string[::-1]):
            if s in "XYZ":
                tmp[Qubit(i)] = ixyz[s]
        term_sum[QubitPauliString(tmp)] = coef
    return QubitPauliOperator(term_sum)

from pytket.partition import measurement_reduction, MeasurementBitMap, MeasurementSetup, PauliPartitionStrat
from pytket.backends.backendresult import BackendResult

# Computing Expectation Values for Pauli-Strings
def compute_expectation_paulistring(
    distribution: dict[tuple[int, ...], float], bitmap: MeasurementBitMap
) -> float:
    value = 0
    for bitstring, probability in distribution.items():
        value += probability * (sum(bitstring[i] for i in bitmap.bits) % 2)
    return ((-1) ** bitmap.invert) * (-2 * value + 1)

# Computing Expectation Values for sums of Pauli-strings multiplied by coefficients
def compute_expectation_value(
    results: list[BackendResult],
    measurement_setup: MeasurementSetup,
    operator: QubitPauliOperator,
) -> float:
    energy = 0
    for pauli_string, bitmaps in measurement_setup.results.items():
        string_coeff = operator.get(pauli_string, 0.0)
        if string_coeff > 0:
            for bm in bitmaps:
                index = bm.circ_index
                distribution = results[index].get_distribution()
                value = compute_expectation_paulistring(distribution, bm)
                energy += complex(value * string_coeff).real
    return energy


#qc is the circuit in the final from that we need
model = QiskitCircuitGeneratorAVQITE(ansatz_filename='ansatz_inp.pkle', incar_filename='../incar')
qc = model.qc
print(qc)
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
obs_list_qiskit = []
const_list = np.zeros(len(obs_list_2D))
pauli_strings_list = []
coefs_list = []

for i, obs_row in enumerate(obs_list_2D):
    pauli_strings = []
    params = []
    for obs in obs_row:
        pauli = "".join([i for i in obs if (i=="I" or i=="X" or i=="Y" or i=="Z")])
        coeff = float("".join([i for i in obs if (i=="-" or i=="." or i.isdigit())]))

        # remove trivial identity
        if pauli != "I"*len(pauli):
            pauli_strings.append(pauli[::-1])
            params.append(coeff)
        else:
            const_list[i] = coeff

    pauli_strings_list.append(pauli_strings)
    coefs_list.append(params)

    obs_list_qiskit.append(SparsePauliOp(pauli_strings, params))

# Computes expectation values using Qiskit StatevectorEstimator
estimator = StatevectorEstimator()
pub = (qc, obs_list_qiskit)
job = estimator.run([pub])
result = job.result()[0]

print("exact expvals:")
for ob, val, vadd in zip(obs_list_qiskit, result.data.evs, const_list):
    # print(f"{ob}: {val+vadd}, {val}")
    print(f"{val+vadd:10.6f}, {val:10.6f}")


if False:
    # noisy simulator
    from qiskit_ibm_runtime import EstimatorV2
    from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

    backend = FakeGuadalupeV2()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)

    estimator = EstimatorV2(mode=backend)

    isa_observables = [op.apply_layout(isa_qc.layout) for op in obs_list_qiskit]

    job = estimator.run([(isa_qc, isa_observables)], precision=0.01)
    result = job.result()[0]
    print("noisy simulator")
    for ob, val, vadd in zip(obs_list_qiskit, result.data.evs, const_list):
            print(f"{val+vadd:10.6f}, {val:10.6f}")

if False:
    # IBM Circuit function
    from qiskit_ibm_catalog import QiskitFunctionsCatalog
    from qiskit_ibm_runtime import QiskitRuntimeService

    catalog = QiskitFunctionsCatalog()
    function = catalog.load("ibm/circuit-function")
    service = QiskitRuntimeService()
    # backend = service.least_busy(operational=True, simulator=False)
    # backend = service.backend("ibm_strasbourg")
    # backend = service.backend("ibm_sherbrooke")
    # backend = service.backend("ibm_fez")
    backend = service.backend("ibm_brisbane")

    print(f"backend: {backend}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)
    isa_observables = [op.apply_layout(isa_qc.layout) for op in obs_list_qiskit]
    pub = (isa_qc, isa_observables)

    for i in range(1):
        job = function.run(
                backend_name=backend.name,
                pubs=[pub],
                options = {"mitigation_level": 1, "default_precision": 0.01},
                )

        print(job)
        print(job.status())


if False:
    # IBM Circuit function
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2


    service = QiskitRuntimeService()

    # backend = service.least_busy(operational=True, simulator=False)
    # backend = service.backend("ibm_strasbourg")
    # backend = service.backend("ibm_sherbrooke")
    # backend = service.backend("ibm_fez")
    # backend = service.backend("ibm_brussels")
    backend = service.backend("ibm_brisbane")


    print(f"backend: {backend}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_qc = pm.run(qc)
    isa_observables = [op.apply_layout(isa_qc.layout) for op in obs_list_qiskit]
    estimator = EstimatorV2(mode=backend, options={"resilience_level": 2})
    job = estimator.run([(isa_qc, isa_observables)], precision=0.01)
    print(job)
    print(job.status())



hub = "ibm-q-howard"
group = "southern-univers"
project = "impurity-model-c"

if True:
    # q-ctrl
    from qiskit_ibm_catalog import QiskitFunctionsCatalog
    catalog = QiskitFunctionsCatalog()
    perf_mgmt = catalog.load("q-ctrl/performance-management")

    qc.measure_all()

    backend_name = "ibm_sherbrooke"
    qctrl_estimator_job = perf_mgmt.run(
        primitive="estimator",
        pubs=[(qc, obs_list_qiskit)],
        instance=hub + "/" + group + "/" + project,
        backend_name=backend_name,
    )
    print(qctrl_estimator_job)
    print(qctrl_estimator_job.status())

