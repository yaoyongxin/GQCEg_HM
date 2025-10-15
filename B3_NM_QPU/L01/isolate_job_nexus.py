import qnexus as qnx
from datetime import datetime
from pathlib import Path


device_name = "H1-1LE"
compile_ref = qnx.filesystem.load(path=Path.cwd()/"compile_ref/0_3")
project = compile_ref.project
circs_c =  qnx.jobs.results(compile_ref)
nshots = 100


results = qnx.execute(
    name=f"execute_job_L01_{datetime.now()}",
    circuits=circs_c,
    n_shots=[nshots]*len(circs_c),
    backend_config=qnx.QuantinuumConfig(device_name=device_name),
    timeout=None,
    project=project,
    )


# job_ref = qnx.start_execute_job(
#         name=f"execute_job_L01_{datetime.now()}",
#         circuits=circs_c,
#         n_shots=[nshots]*len(circs_c),
#         backend_config=qnx.QuantinuumConfig(device_name=device_name),
#         project=project,
#         )

