This package is used to access SuperstaQ via a Web API through [Cirq](https://github.com/quantumlib/Cirq).
Cirq programmers can take advantage of the applications, pulse level optimization, and write-once-target-all
features of SuperstaQ with this package.


Please note that Python version `3.7` or higher is required. cirq-superstaq and all of its
dependencies can be installed via:

```
git clone https://github.com/SupertechLabs/qiskit-superstaq.git
cd qiskit-superstaq
conda create -n cirq-superstaq-env python=3.8
conda activate cirq-superstaq-env
pip install -r dev-requirements.txt
pip install -e .
```

Make sure to always be in the cirq-superstaq-env (via ``conda activate cirq-superstaq-env``) when you're working on cirq-superstaq.

### Creating and submitting a circuit through cirq-superstaq
```
import cirq

import cirq_superstaq


q0 = cirq.LineQubit(0)
q1 = cirq.LineQubit(1)

circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0))


service = cirq_superstaq.Service(
    api_key="""Insert superstaq token that you received from https://superstaq.super.tech""",
    verbose=True,
)

job = service.create_job(circuit=circuit, repetitions=1, target="ibmq_qasm_simulator")
print("This is the job that's created ", job.status())

print(job.counts())

```
