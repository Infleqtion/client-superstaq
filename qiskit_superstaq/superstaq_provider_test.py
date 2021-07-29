import qiskit_superstaq as qss


def test_provider() -> None:
    ss_provider = qss.superstaq_provider.SuperstaQProvider(access_token="MY_TOKEN")

    assert str(ss_provider.get_backend("ibmq_qasm_simulator")) == str(
        qss.superstaq_backend.SuperstaQBackend(
            provider=ss_provider,
            url=qss.API_URL,
            backend="ibmq_qasm_simulator",
        )
    )

    assert str(ss_provider) == "<SuperstaQProvider(name=superstaq_provider)>"

    assert (
        repr(ss_provider) == "<SuperstaQProvider(name=superstaq_provider, access_token=MY_TOKEN)>"
    )

    backend_names = [
        "aqt_device",
        "ionq_device",
        "rigetti_device",
        "ibmq_botoga",
        "ibmq_casablanca",
        "ibmq_jakarta",
        "ibmq_qasm_simulator",
    ]

    backends = []
    for name in backend_names:
        backends.append(
            qss.superstaq_backend.SuperstaQBackend(
                provider=ss_provider, url=qss.API_URL, backend=name
            )
        )

    assert ss_provider.backends() == backends
