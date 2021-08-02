import json
import os

import cirq
import requests

import cirq_superstaq as css


def _get_api_url() -> str:
    return os.getenv("SUPERSTAQ_REMOTE_HOST", default=css.API_URL)


def _get_headers() -> dict:
    return {
        "Authorization": os.environ["SUPERSTAQ_API_KEY"],
        "Content-Type": "application/json",
    }


def _should_verify_requests() -> bool:
    """Returns the appropriate ``verify`` kwarg for requests.

    When we test locally, we don't have a certificate so we can't verify.
    When running against the production server (css.API_URL), we should verify.
    """
    return _get_api_url() == css.API_URL


def aqt_compile(circuit: cirq.Circuit) -> cirq.Circuit:
    """Compiles the given circuit to the Berkeley-AQT device.

    Args:
        circuit: a cirq Circuit object with operations on qubits 4 through 8.
    Returns:
        an optimized cirq Circuit that is decomposed to AQT's native gate set (RZ, X90, CZ).
    """

    superstaq_json = {"circuit": json.loads(cirq.to_json(circuit))}

    result = requests.post(
        _get_api_url() + "/" + css.API_VERSION + "/aqt_compile",
        json=superstaq_json,
        headers=_get_headers(),
        verify=_should_verify_requests(),
    )

    result.raise_for_status()
    response = result.json()

    out_circuit = cirq.read_json(json_text=response["compiled_circuit"])
    return out_circuit
