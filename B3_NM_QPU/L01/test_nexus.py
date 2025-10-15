from datetime import datetime

import qnexus as qnx

from pytket import Circuit

my_job_name_prefix = datetime.now()

my_project_ref = qnx.projects.get_or_create(name="My test Project")

my_circuit_ref = qnx.circuits.upload(
    name=f"My Circuit from {datetime.now()}",
    circuit = Circuit(2).H(0).CX(0,1).measure_all(),
    project = my_project_ref,
)

# Compile the circuit (blocking), to receive a list of compiled CircuitRefs

compiled_circuits = qnx.compile(
    circuits=[my_circuit_ref],
    name=f"{my_job_name_prefix}_compile",
    optimisation_level=1,
    backend_config=qnx.QuantinuumConfig(device_name="H1-1LE"),
    project=my_project_ref,
)

compiled_circuits.df()

# Execute the circuit (blocking), to receive a list of pytket BackendResults

results = qnx.execute(
    circuits=compiled_circuits,
    name=f"{my_job_name_prefix}_execute",
    n_shots=[100]* len(compiled_circuits),
    backend_config=qnx.QuantinuumConfig(device_name="H1-1LE"),
    project=my_project_ref,
)

print(results[0].get_counts())
