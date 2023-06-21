from typing import Dict, List, Tuple
from datetime import datetime

import cirq_superstaq as css
import supermarq

BENCHMARKS: List[Tuple[supermarq.benchmark.Benchmark, str]] = [
    (supermarq.ghz.GHZ(5), "ghz5"),
    (supermarq.hamiltonian_simulation.HamiltonianSimulation(4), "hsim4"),
    (supermarq.mermin_bell.MerminBell(3), "mb3"),
    (supermarq.bit_code.BitCode(3, 3, [1, 0, 1]), "bitcode3"),
]

def get_qpu_targets(targets: Dict[str, str]) -> List[str]:
    qpu_targets = []
    run_targets = targets.get("compile-and-run")
    for t in run_targets:
        if t.endswith("qpu"):
            qpu_targets.append(t)
    return qpu_targets

service = css.Service()
targets = service.get_targets()
qpu_targets = get_qpu_targets(targets)

for target in qpu_targets:
    for benchmark, label in BENCHMARKS:
        date = datetime.today().strftime('%Y-%m-%d')
        tag = f"{date}-{label}-{target}"
        print(tag)

        # Since there is no way to determine if a target supports mid-circuit measurement purely 
        # through `service.target_info()`, we use a `try` block to catch when we submit an invalid
        # circuit to a device that does not support it. 
        try:
            job = service.create_job(
                benchmark.circuit(), repetitions=1000, target=target, options={"tag": tag, "qiskit_pulse": False}
            )
        except Exception:
            pass