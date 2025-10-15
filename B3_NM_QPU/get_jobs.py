from qiskit_ibm_catalog import QiskitFunctionsCatalog



catalog = QiskitFunctionsCatalog()
jobs = catalog.jobs()
for job in jobs:
    print(job.job_id, job.status())
