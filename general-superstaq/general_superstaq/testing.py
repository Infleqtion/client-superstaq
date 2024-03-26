from general_superstaq.typing import Target

TARGET_LIST = {
    "aqt_keysight_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "aqt_zurich_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "aws_dm1_simulator": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "aws_sv1_simulator": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "aws_tn1_simulator": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "cq_hilbert_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "cq_hilbert_simulator": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ibmq_brisbane_qpu": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ibmq_extended-stabilizer_simulator": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ibmq_fake-athens_qpu": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ibmq_fake-lima_qpu": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ibmq_kyoto_qpu": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ibmq_mps_simulator": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ibmq_qasm_simulator": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ibmq_stabilizer_simulator": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ibmq_statevector_simulator": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ionq_aria-1_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": False,
    },
    "ionq_aria-2_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": False,
    },
    "ionq_forte-1_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": False,
    },
    "ionq_harmony_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": False,
    },
    "ionq_ion_simulator": {
        "supports_submit": True,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "oxford_lucy_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": False,
    },
    "qtm_h1-1_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "qtm_h1-1e_simulator": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "qtm_h2-1_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "rigetti_aspen-10_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": True,
    },
    "rigetti_aspen-11_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": True,
    },
    "rigetti_aspen-8_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": True,
    },
    "rigetti_aspen-9_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": True,
    },
    "rigetti_aspen-m-1_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": True,
    },
    "rigetti_aspen-m-2_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": True,
    },
    "rigetti_aspen-m-3_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": False,
        "retired": False,
    },
    "qscout_peregrine_qpu": {
        "supports_submit": False,
        "supports_submit_qubo": False,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
    "ss_unconstrained_simulator": {
        "supports_submit": True,
        "supports_submit_qubo": True,
        "supports_compile": True,
        "available": True,
        "retired": False,
    },
}

RETURNED_TARGETS = [
    Target(target=target_name, **properties) for target_name, properties in TARGET_LIST.items()
]
