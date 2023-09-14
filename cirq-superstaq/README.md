# `cirq-superstaq`

![cirq-superstaq's default workflow](https://github.com/Infleqtion/client-superstaq/actions/workflows/ci.yml/badge.svg)

This package is used to access Superstaq via a Web API through [Cirq](https://github.com/quantumlib/Cirq).
Cirq programmers can take advantage of the applications, pulse level optimizations, and write-once-target-all
features of Superstaq with this package.

`cirq-superstaq` is [available on PyPI](https://pypi.org/project/cirq-superstaq) and can be installed with:

```
pip install cirq-superstaq
```

Please note that Python version `3.8` or higher is required. For further installation instructions, see [here](https://github.com/Infleqtion/client-superstaq#readme).

### Creating and submitting a circuit through cirq-superstaq
```python
import cirq
import cirq_superstaq as css

q0 = cirq.LineQubit(0)
q1 = cirq.LineQubit(1)

circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0))

service = css.Service(
    api_key="""Insert superstaq token that you received from https://superstaq.infleqtion.com""",
    verbose=True,
)

# Submitting a circuit to "ibmq_qasm_simulator". Providing the "dry-run" method parameter instructs Superstaq to simulate the circuit, and is available to free trial users.
job = service.create_job(circuit, repetitions=1, target="ibmq_qasm_simulator", method="dry-run")
print("This is the job that's created ", job.status())
print(job.counts())
```
