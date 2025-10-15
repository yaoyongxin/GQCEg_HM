from qiskit import QuantumCircuit
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[127];
x q[10];
rz(-pi/2) q[45];
sx q[45];
rz(-0.8785596450773188) q[45];
sx q[54];
rz(-pi/2) q[54];
ecr q[45],q[54];
rz(-pi) q[54];
sx q[54];
rz(2.0382389591034027) q[54];
sx q[54];
ecr q[45],q[54];
rz(0.8785596450773188) q[45];
sx q[45];
rz(-pi/2) q[45];
rz(-pi/2) q[54];
sx q[54];
rz(-pi/2) q[78];
sx q[78];
rz(-1.6099524488118009) q[78];
rz(-pi) q[79];
sx q[79];
rz(-pi/2) q[79];
ecr q[79],q[78];
rz(-pi) q[78];
sx q[78];
rz(2.0382389591034027) q[78];
sx q[78];
ecr q[79],q[78];
rz(1.609952448811799) q[78];
sx q[78];
rz(-pi/2) q[78];
rz(-pi/2) q[79];
sx q[79];
rz(-pi) q[79];
x q[88];
"""
circuit = QuantumCircuit.from_qasm_str(qasm)
print(circuit.depth())
print(circuit.count_ops())
