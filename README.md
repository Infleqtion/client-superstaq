This package is used to access SuperstaQ via a Web API through [Qiskit](https://qiskit.org/). Qiskit programmers
can take advantage of the applications, pulse level optimizations, and write-once-target-all
features of SuperstaQ with this package.


Please note that Python version `3.7` or higher is required. qiskit-superstaq and all of its
dependencies can be installed via:

```
python3 -m venv venv_qiskit_superstaq
source venv_qiskit_superstaq/bin/activate
pip install qiskit-superstaq
pip install -e .

# Run the following to install neutral atom device dependencies.
pip install -r neutral-atom-requirements.txt
```

### Creating and submitting a circuit through qiskit-superstaq
```python3

import qiskit
import qiskit_superstaq

token = "Insert superstaq token that you received from https://superstaq.super.tech"

superstaq = qiskit_superstaq.superstaq_provider.SuperstaQProvider(
    token,
    url=qiskit_superstaq.API_URL,
)

backend = superstaq.get_backend("ibmq_qasm_simulator")
qc = qiskit.QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure(0, 0)
qc.measure(1, 1)

print(qc)
job = backend.run(qc, shots=100)
print(job.result().get_counts())
```
