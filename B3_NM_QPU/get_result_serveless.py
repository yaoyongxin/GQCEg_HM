from qiskit_ibm_catalog import QiskitServerless
serverless = QiskitServerless()


def get_job(jobs, jid):
    for job in jobs:
        if str(job) == jid:
            return job
    return None


# 03aa1879-17b8-4449-ac55-8c2497604e86 ERROR
# 96f8be32-0583-4155-b0e0-958eea549bcd DONE
# a8ea42e5-e1e8-45e0-ae2c-d11fc2510f90 ERROR
# f8d1fed9-7333-4a64-8da4-029161b02ebd ERROR
# 6201aba5-ad18-4396-b339-449767e24c1c DONE
# 5b5f3761-3f72-44c4-a179-6ad8fa58fa8b DONE
# 1445dbd9-f645-4ccd-8c62-e3b5ee5d5bbb DONE
# b061b9ac-d399-49d6-aaf2-05906c6a68d4 DONE
# 4d22d712-0116-4c53-9274-2f1b93dda2f9 ERROR
# 9046b08e-4f23-4aef-801f-6c7c353b7e7b DONE


# jid = "<Job | 1188f259-2d87-4fe1-aa38-185b1e3f3e03>"    # L10, q-ctrl, 1-obs
# jid = "<Job | b5ac3f7f-6118-49bb-b5f4-897873f27f41>"    # L10, q-ctrl
# jid = "<Job | 8c7cc671-a48e-41be-9e5f-61f1f5445944>"    # L10, q-ctrl
# jid = "<Job | b822473b-4d7a-4e89-8bf9-4c0e3a187406>"    # L30, q-ctrl
# jid = "<Job | 0d25436a-3bf1-4740-aa2d-b08b15daf70b>"    # L30, q-ctrl
# jid = "<Job | 57f7f93c-686b-4afc-8a21-7403ab27ff9c>"    # L30, q-ctrl
# jid = "<Job | ef80ddd7-2689-4922-a19a-bb7b12c08871>"    # L30, q-ctrl, no measure-all
# jid = "<Job | 715e4ce3-d2c3-43d9-b3b5-d86d3f24535c>"    # L30, q-ctrl, 2**16 shots
# jid = "<Job | c0666eb1-b6b3-4970-90f0-2b705e5f693e>"    # L50, q-ctrl
# jid = "<Job | 9046b08e-4f23-4aef-801f-6c7c353b7e7b>"    # test_algorithmiq_tem
# jid = "<Job | b061b9ac-d399-49d6-aaf2-05906c6a68d4>"    # L10, algorithmiq tem, unique_layers=4  (bug: pauli reversed.)
# jid = "<Job | 1445dbd9-f645-4ccd-8c62-e3b5ee5d5bbb>"    # L10, algorithmiq tem, unique-layers=20 (bug: pauli reversed.)
# jid = "<Job | 5b5f3761-3f72-44c4-a179-6ad8fa58fa8b>"    # L10, algorithmiq tem, unique-layers=20
# jid = "<Job | 6201aba5-ad18-4396-b339-449767e24c1c>"    # L10, algorithmiq tem, unique-layers=20
# jid = "<Job | f8d1fed9-7333-4a64-8da4-029161b02ebd>"    # L30, algorithmiq tem, unique-layers=20, opt-3, (1,2), (2,1) connection
# jid = "<Job | a8ea42e5-e1e8-45e0-ae2c-d11fc2510f90>"    # L30, algorithmiq tem, unique-layers=20, opt-1, (1,2) connection

# jid = "<Job | 6a564c11-b0b6-45c6-ba73-1f1f7c9775da>"    # L30, IBM circuit function ml=1
# jid = "<Job | 2f815018-b5aa-4d8f-abea-82b597de5d74>"    # L30, IBM circuit function ml=1
# jid = "<Job | 6529fa53-135d-46f0-b75f-18ad89b65d5a>"    # L30, IBM circuit function ml=1
#
# jid = "<Job | 5cabcaa1-5ea6-4606-b429-6724b1bdfef5>"    # L30, IBM circuit function ml=2, error
# jid = "<Job | 93e9fae1-cecd-4cb6-b809-716a2be0c67c>"    # L30, IBM circuit function ml=2
# jid = "<Job | 58a50932-ef77-4d74-9783-093a4c03c20a>"    # L30, IBM circuit function ml=2
# jid = "<Job | d17232d5-9585-4fec-b76d-477b1b2b218a>"    # L30, IBM circuit function ml=2
# jid = "<Job | 33413ffe-5fbf-46aa-8eb7-c2a921883dd4>"    # L30, IBM circuit function ml=2

# jid = "<Job | c8bd7458-1b6f-48b8-87e9-cc308d6011d3>"    # L30, IBM circuit function ml=3
jid = "<Job | a2947558-afac-4bc3-aec6-ab0e5fa693a6>"    # L30, IBM circuit function ml=3
jid = "<Job | 824a14f8-6735-42f1-ab5d-dabba3bd2c98>"    # L30, IBM circuit function ml=3


job = get_job(serverless.jobs(), jid)
print(job.status())

job_result = job.result()

print(job_result[0].data.evs)

for idx, eval in enumerate(job_result[0].data.evs):
    print(f"{idx}: {eval:.6f} std: {job_result[0].data.stds[idx]:.6f}")

try:
    print(job_result[0].data.evs_noise_factors)
except:
    pass
