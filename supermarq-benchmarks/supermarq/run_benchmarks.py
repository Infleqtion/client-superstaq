from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import cirq
import cirq_superstaq as css

import supermarq

if TYPE_CHECKING:
    from general_superstaq.typing import Target


BENCHMARKS: list[tuple[supermarq.benchmark.Benchmark, str]] = [
    (supermarq.ghz.GHZ(5), "ghz5"),
    (supermarq.hamiltonian_simulation.HamiltonianSimulation(4), "hsim4"),
    (supermarq.mermin_bell.MerminBell(3), "mb3"),
    (supermarq.bit_code.BitCode(3, 3, [1, 0, 1]), "bitcode3"),
]


def get_qpu_targets(target_list: list[Target]) -> list[Target]:
    """Gets real device targets.

    Args:
        target_list: Output from `service.get_targets()`.

    Returns:
        A list with the targets corresponding to real devices.
    """
    qpu_targets = [
        target_info
        for target_info in target_list
        if target_info.target.endswith("qpu")
        and not target_info.target.startswith("ionq")
        and not target_info.target.startswith("rigetti")
    ]
    return qpu_targets


if __name__ == "__main__":
    service = css.Service()
    targets = service.get_targets()
    qpu_targets = get_qpu_targets(targets)

    for target in qpu_targets:
        for benchmark, label in BENCHMARKS:
            date = datetime.today().strftime("%Y-%m-%d")
            tag = f"{date}-{label}-{target}"

            target_info = service.target_info(target.target)
            if label == "bitcode3" and not target_info.get("supports_midcircuit_measurement"):
                continue

            try:
                circuit = benchmark.circuit()
                assert isinstance(circuit, cirq.Circuit)
                job = service.create_job(
                    circuit,
                    repetitions=1000,
                    target=target.target,
                    tag=tag,
                    lifespan=3653,
                )
            except Exception:
                print(f"{label} on {target.target} failed.")
