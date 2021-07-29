import cirq
import cirq_superstaq


q0 = cirq.LineQubit(0)
q1 = cirq.LineQubit(1)

circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0))


service = cirq_superstaq.Service(
    remote_host="https://127.0.0.1:5000",
    api_key="""insert superstaq token""",
    verbose=True,
)


# service = cirq_superstaq.Service(remote_host="https://127.0.0.1:5000",
# api_key="ya29.a0ARrdaM8fpfWYy4ckfhV--f7zwtDrs9GXRkRCTgiwNJg9cxvBJbWx_BP9bjXnon8iWZokpeXdmTi0
# dixrwyvVaYt7tSrmb76KCOWDygZewm0VcD5RuMTRKVlm06RgoD8Y2u8VO3Wl4luKTB-MXCIYHZJLALCF7w")

job = service.create_job(circuit=circuit, repetitions=1, target="ibmq_qasm_simulator")
print("This is the job that's created ", job.status())
# job = service.get_job("60e86491b5880821786cca7a")

print(job.counts())
