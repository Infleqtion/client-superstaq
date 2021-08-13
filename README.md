This package is used to access SuperstaQ via a Web API through [Cirq](https://github.com/quantumlib/Cirq).
Cirq programmers can take advantage of the applications, pulse level optimization, and write-once-target-all
features of SuperstaQ.


Please note that Python version `3.7` or higher is required. cirq-superstaq and all of its
dependencies can be installed via:

```
pip install cirq-superstaq
cd cirq-superstaq
python3 -m venv venv_cirq_superstaq
source venv_cirq_superstaq/bin/activate
pip install -r dev-requirements.txt
pip install -e .
```

Make sure to always be in the cirq-superstaq-env (via ``source venv_cirq_superstaq/bin/activate``) when you're working on cirq-superstaq.

### Creating and submitting a circuit through cirq-superstaq
```
python3

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
