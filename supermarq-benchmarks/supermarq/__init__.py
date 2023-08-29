from . import benchmark, converters, features, plotting, simulation, stabilizers
from ._version import __version__
from .benchmarks import (
    bacon_shor_code,
    bit_code,
    ghz,
    hamiltonian_simulation,
    mermin_bell,
    phase_code,
    qaoa_fermionic_swap_proxy,
    qaoa_vanilla_proxy,
    surface_code,
    vqe_proxy,
)

__all__ = [
    "__version__",
    "bacon_shor_code",
    "benchmark",
    "bit_code",
    "converters",
    "features",
    "ghz",
    "hamiltonian_simulation",
    "mermin_bell",
    "phase_code",
    "plotting",
    "qaoa_fermionic_swap_proxy",
    "qaoa_vanilla_proxy",
    "simulation",
    "stabilizers",
    "surface_code",
    "vqe_proxy",
]
