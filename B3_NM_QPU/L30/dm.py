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
        if abs(string_coeff) > 0:
            for bm in bitmaps:
                index = bm.circ_index
                distribution = results[index].get_distribution()
                value = compute_expectation_paulistring(distribution, bm)
                energy += complex(value * string_coeff).real
    return energy


#qc is the circuit in the final from that we need
model = QiskitCircuitGeneratorAVQITE(ansatz_filename='ansatz_inp.pkle', incar_filename='../incar')
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

    obs_list_qiskit.append([SparsePauliOp(pauli_strings, params)])

# Computes expectation values using Qiskit StatevectorEstimator
estimator = StatevectorEstimator()
pub = (qc, obs_list_qiskit)
job = estimator.run([pub])
result = job.result()[0]

for ob, val, vadd in zip(obs_list_qiskit, result.data.evs, const_list):
    # print(f"{ob}: {val+vadd}, {val}")
    print(f"{val[0]+vadd:10.6f}, {val[0]:10.6f}")
dm_ref = result.data.evs[:, 0]

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

    run_mode = ["upload", "compile", "execute", "analysis"][3]

    device_name = ["H1-1LE", "H1-Emulator", "H1-1E", "H1-1"][3]

    nshots = 1000

    input(f"{run_mode} at {device_name} eith {nshots} shots. OK?")

    # Nexus quantinuum
    from pytket.extensions.qiskit import qiskit_to_tk
    from pathlib import Path

    qc = qiskit_to_tk(qc)

    # for command in qc:
    #     print(command)

    # get observables in pytket
    obs_list_pytket = [pauliStringsToQubitPauliOperator(pauli_strings, coeffs)
            for pauli_strings, coeffs in zip(pauli_strings_list, coefs_list)]

    import qnexus as qnx
    from datetime import datetime
    # qnx.login()
    project = qnx.projects.get(name_like="gGAHubbard")

    # measurement
    terms = []
    for op in obs_list_pytket:
        terms += [term for term in op._dict.keys()]
    terms = list(set(terms))
    measurement_setup = measurement_reduction(
            terms, strat=PauliPartitionStrat.CommutingSets
            )

    # add measurement circuits
    if run_mode == "upload":
        circuit_list = []
        for mc in measurement_setup.measurement_circs:
            c = qc.copy()
            c.append(mc)
            # Upload each measurement circuit to Nexus with correct params
            measurement_circuit_ref = qnx.circuits.upload(
                circuit=c,
                name=f"state prep + meas circuit_L30 from {datetime.now()}",
                project = project,
            )
            circuit_list.append(measurement_circuit_ref)

        # save upload refs.
        for i, cref in enumerate(circuit_list):
            qnx.filesystem.save(
                    ref=cref,
                    path=Path.cwd() / "upload_ref" / f"{i}",
                    mkdir=True,
                    )
        quit()


    # retrieve uploaded circuit_list
    if run_mode == "compile":
        circuit_list = []
        for i in range(4):
            cref = qnx.filesystem.load(
                    path=Path.cwd() / "upload_ref" / f"{i}",
                    )
            circuit_list.append(cref)

        # asyncronous manner
        circs_c = qnx.start_compile_job(
            circuits=circuit_list,
            name=f"Compile Job_L30_8q from {datetime.now()}",
            optimisation_level=2,
            backend_config=qnx.QuantinuumConfig(device_name=device_name),
            project=project,
            )

        # save compile job ref
        qnx.filesystem.save(
                ref=circs_c,
                path=Path.cwd() / "compile_ref" / f"0_3_{device_name}",
                mkdir=True,
                )
        quit()

    if run_mode == "execute":
        # retrieve compile circuits
        compile_ref = qnx.filesystem.load(
                path=Path.cwd() / "compile_ref" / f"0_3_{device_name}",
                )

        circs_c =  [res.get_output() for res in qnx.jobs.results(compile_ref)]

        # Execute circuits with Nexus

        job_ref = qnx.start_execute_job(
                name=f"execute_job_L30_8q_{datetime.now()}",
                circuits=circs_c,
                n_shots=[nshots]*len(circs_c),
                backend_config=qnx.QuantinuumConfig(
                        device_name=device_name,
                        noisy_simulation=False,
                        ),
                project=project,
                )

        # save exec job ref
        qnx.filesystem.save(
                ref=job_ref,
                path=Path.cwd() / "exec_ref" / f"0_3_{device_name}",
                mkdir=True,
                )

        quit()

    # retrieve results
    exec_ref = qnx.filesystem.load(
            path=Path.cwd() / "exec_ref" / f"0_3_{device_name}_run1",
            )
    print(exec_ref.df())

    results = [res.download_result() for res in qnx.jobs.results(exec_ref)]

    for i, op in enumerate(obs_list_pytket):
        res = compute_expectation_value(
                results, measurement_setup, op
                )
        print(f"nexus: {i} {res.real:10.6f} vs ref: {dm_ref[i]:10.6f}")
