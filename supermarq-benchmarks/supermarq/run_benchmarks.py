from datetime import datetime
from typing import Dict, List, Tuple

import cirq
import cirq_superstaq as css

import supermarq

BENCHMARKS: List[Tuple[supermarq.benchmark.Benchmark, str]] = [
    (supermarq.ghz.GHZ(5), "ghz5"),
    (supermarq.hamiltonian_simulation.HamiltonianSimulation(4), "hsim4"),
    (supermarq.mermin_bell.MerminBell(3), "mb3"),
    (supermarq.bit_code.BitCode(3, 3, [1, 0, 1]), "bitcode3"),
]


def get_qpu_targets(targets: Dict[str, List[str]]) -> List[str]:
    """Get real device targets.

    Args:
        targets: Output from `service.get_targets()`.

    Returns:
        A list with the targets corresponding to real devices.
    """
    qpu_targets: List[str] = []
    run_targets = targets.get("compile-and-run", [])

    for t in run_targets:
        if t.endswith("qpu") and not t.startswith("ionq") and not t.startswith("rigetti"):
            qpu_targets.append(t)
    return qpu_targets


if __name__ == "__main__":
    service = css.Service()
    targets = service.get_targets()
    qpu_targets = get_qpu_targets(targets)

    for target in qpu_targets:
        for benchmark, label in BENCHMARKS:
            date = datetime.today().strftime("%Y-%m-%d")
            tag = f"{date}-{label}-{target}"

            target_info = service.target_info(target)
            if label == "bitcode3" and not target_info.get("supports_midcircuit_measurement"):
                continue

            try:
                circuit = benchmark.circuit()
                assert isinstance(circuit, cirq.Circuit)
                job = service.create_job(
                    circuit,
                    repetitions=1000,
                    target=target,
                    tag=tag,
                    lifespan=3653,
                )
            except Exception:
                print(f"{label} on {target} failed.")
