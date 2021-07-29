import qiskit
import qiskit_superstaq as qss

token = "insert-token-here"

superstaq = qss.superstaq_provider.SuperstaQProvider(
    token,
    url=qss.API_URL,
)

print(superstaq.get_backend("ibmq_qasm_simulator"))

backend = superstaq.get_backend("ibmq_qasm_simulator")
qc = qiskit.QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure(0, 0)
qc.measure(1, 1)

print(qc)
job = backend.run(qc, shots=100)
print(job.result().get_counts())


# qc = QuantumCircuit(5, 5)
# qc.x(0)
# qc.swap(0, 1)
# qc.swap(1, 2)
# qc.swap(2, 3)
# qc.swap(3, 4)
# qc.measure(0, 0)
# qc.measure(1, 1)
# qc.measure(2, 2)
# qc.measure(3, 3)
# qc.measure(4, 4)

# job = backend.run(qc, shots=10, target="ibmq_qasm_simulator")
# job = SuperstaQJob(backend, "60d26ea74fc8d411cfc97269")

# print(job.get_counts())
