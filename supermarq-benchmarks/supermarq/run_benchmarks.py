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
    run_targets = targets.get("compile-and-run")

    if run_targets is None:
        return qpu_targets

    for t in run_targets:
        if t.endswith("qpu"):
            qpu_targets.append(t)
    return qpu_targets


service = css.Service()
targets = service.get_targets()
qpu_targets = get_qpu_targets(targets)

for target in qpu_targets:
    for benchmark, label in BENCHMARKS:
        date = datetime.today().strftime("%Y-%m-%d")
        tag = f"{date}-{label}-{target}"

        # Since there is no way to determine if a target supports mid-circuit measurement purely
        # through `service.target_info()`, we use a `try` block to catch when we submit an invalid
        # circuit to a device that does not support it.
        # Related issue: https://github.com/Infleqtion/client-superstaq/issues/550.

        try:
            circuit = benchmark.circuit()
            if isinstance(circuit, cirq.Circuit):
                job = service.create_job(
                    circuit,
                    repetitions=1000,
                    target=target,
                    tag=tag,
                    lifespan=3653,
                )
            else:
                for idx, c in enumerate(circuit):
                    job = service.create_job(
                        c,
                        repetitions=1000,
                        target=target,
                        tag=f"{tag}-{idx}",
                        lifespan=3653,
                    )
        except Exception:
            pass
