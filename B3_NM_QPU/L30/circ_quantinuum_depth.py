import json

# Load the circuit JSON file
with open('circuit.json', 'r') as f:
    circuit_data = json.load(f)

# Extract the list of operations
operations = circuit_data.get('operations', [])

depth = 0
qubits = []
num2 = 0
for cmd in circuit_data["commands"]:
    for _, q in cmd["args"]:
        if q in qubits:
            depth += 1
            qubits = [q]
        else:
            qubits.append(q)
    if cmd['op']['type'] == 'ZZPhase':
        num2 += 1

if len(qubits) > 0:
    depth += 1

print(f"total gates: {len(operations)} Circuit depth: {depth} zz gates: {num2}")

