"""
Creates and simulates a bell circuit.

Example Output:

job has successfully run
{'00': 52, '11': 48}

"""

import qiskit

import qiskit_superstaq as qss

# SuperstaQ token retrieved through API
token = "insert api token"

# Create provider using authorization token
superstaq = qss.SuperstaQProvider(token)

# Retrieve backend from superstaq provider
backend = superstaq.get_backend("ibmq_qasm_simulator")

# Standard bell circuit
qc = qiskit.QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Submit job to backend
job = backend.run(qc, shots=100)

# Get job status
print(job.status().value)

# Get result of job
result = job.result()

# Print job counts
print(result.get_counts())
