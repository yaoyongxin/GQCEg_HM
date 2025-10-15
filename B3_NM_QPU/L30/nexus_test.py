from datetime import datetime
from pytket import Circuit
import qnexus as qnx


# qnx.login()
project = qnx.projects.get(name_like="gGAHubbard")

circuit = Circuit(2).H(0).CX(0,1).measure_all()

my_circuit = qnx.circuits.upload(
    name=f"Test Circuit from {datetime.now()}",
    circuit = circuit,
    project = project,
)

compiled_circuits = qnx.compile(
    circuits=[my_circuit],
    name=f"Test Compile Job from {datetime.now()}",
    optimisation_level=1,
    backend_config=qnx.QuantinuumConfig(device_name="H1-1LE"),
    project=project,
)

print(compiled_circuits)
