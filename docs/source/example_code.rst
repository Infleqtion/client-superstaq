Example Code
============
Below are two examples, one using ``cirq-superstaq`` and another using ``qiskit-superstaq``.

An Example in Cirq
------------------
Here we show an example to create and submit a circuit through ``cirq-superstaq``.

First, have your API token from https://superstaq.super.tech ready (see `Accessing Credentials <credentials.html>`_ for more info). Then, import the required packages

.. code-block:: python

    import cirq
    import cirq_superstaq as css

Create your circuit using ``Cirq``.

.. code-block:: python

    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0))

    service = css.Service(
        api_key="""Insert superstaq token that you received from https://superstaq.super.tech""",
        verbose=True,
    )

Submit your circuit.

.. code-block:: python

    job = service.create_job(circuit=circuit, repetitions=1, target="ibmq_qasm_simulator")
    print("This is the job that's created ", job.status())
    print(job.counts())

An Example in Qiskit
--------------------
Here we show an example to create and submit a circuit through ``qiskit-superstaq``.

First, have your API token from https://superstaq.super.tech ready (see `Accessing Credentials <credentials.html>`_ for more info). Then, import the required packages and set your token.

.. code-block:: python

    import qiskit
    import qiskit_superstaq as qss

    token = "<API token that you received from https://superstaq.super.tech>"

    superstaq = qss.superstaq_provider.SuperstaQProvider(
        token,
        remote_host=qss.API_URL,
    )

Define your backend and create your circuit using ``Qiskit``.

.. code-block:: python

    backend = superstaq.get_backend("ibmq_qasm_simulator")
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)

Submit your circuit.

.. code-block:: python

    print(qc)
    job = backend.run(qc, shots=100)
    print(job.result().get_counts())