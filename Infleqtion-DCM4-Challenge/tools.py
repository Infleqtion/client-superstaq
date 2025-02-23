"""Tools for the DCM4 Challenge air defense model demo"""

import numpy as np
import dimod
import cirq

import networkx as nx
import matplotlib.pyplot as plt
import itertools
from collections.abc import Sequence
import sympy

from IPython.display import clear_output
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import scipy

def JACOBIAN(x):
    """Rosenbrock Jacobian. See https://docs.scipy.org/doc/scipy/tutorial/optimize.html"""
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


def HESSIAN(x):
    """Rosenbrock Hessian. See https://docs.scipy.org/doc/scipy/tutorial/optimize.html."""
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H

class AirDefenceModel:
    def __init__(
        self,
        asset_values,
        asset_destruction_probs,
        target_destruction_probs,
        weapon_amunition_limits,
    ):
        self.asset_values = asset_values
        self.asset_destruction_probs = asset_destruction_probs
        self.target_destruction_probs = target_destruction_probs
        self.weapon_amunition_limits = weapon_amunition_limits

        self.num_targets = target_destruction_probs.shape[1]
        self.num_weapons = target_destruction_probs.shape[0]
        self.num_assets = len(asset_values)

        self.T = np.arange(self.num_targets)
        self.W = np.arange(self.num_weapons)
        self.K = np.arange(self.num_assets)

        self.J_matrix, self.h_vec, self.offset, self.bqm_vars, self.exact_solution = (
            self._build_qubo()
        )

        self.qubits = cirq.LineQubit.range(len(self.bqm_vars))

        self.qaoa_circuit = self.build_circuit()

        self.sim = cirq.Simulator()

        self._results = []

    @property
    def num_vars(self):
        return len(self.bqm_vars)

    def _build_qubo(self):
        cqm = dimod.ConstrainedQuadraticModel()

        # Create variables
        X = np.array([[dimod.Binary(f"X[{i},{j}]") for j in self.T] for i in self.W])
        X_labels = np.array([[f"X[{i},{j}]" for j in self.T] for i in self.W])

        # Add objective - only need the x-dependent terms, drop the overall constant! Also rescale.
        objective = sum(
            sum(
                self.asset_destruction_probs[j, k]
                * sum(X[i, j] * self.target_destruction_probs[i, j] for i in self.W)
                for j in self.T
            )
            for k in self.K
        )  # /(np.min(asset_destruction_probabilities) * np.min(target_destruction_probabilities))
        cqm.set_objective(-objective)  # Include minus sign to minimise

        for i in self.W:
            cqm.add_constraint(
                sum(X[i, j] for j in self.T) == self.weapon_amunition_limits[i],
            )

        # Convert to unconstrained model
        bqm, _ = dimod.cqm_to_bqm(cqm)

        linear, quadratic, offset = bqm.to_ising()

        bqm_vars = list(bqm.variables)
        num_bqm_vars = len(bqm_vars)

        J_matrix = np.zeros((num_bqm_vars, num_bqm_vars))
        for i, a in enumerate(X_labels.flatten()):
            for j, b in enumerate(X_labels.flatten()):
                if (a, b) in quadratic.keys():
                    J_matrix[i, j] = quadratic[(a, b)]

        h_vec = np.zeros(num_bqm_vars)
        for i, x in enumerate(X_labels.flatten()):
            if x in linear.keys():
                h_vec[i] = linear[x]

        exact_solution = (
            dimod.ExactCQMSolver()
            .sample_cqm(cqm)
            .filter(lambda d: d.is_feasible)
            .lowest()
            .record.sample.flatten()
            .reshape(self.num_weapons, self.num_targets)
        )

        return (J_matrix, h_vec, offset, bqm_vars, exact_solution)

    def plot_solution(self, solution_array) -> None:

        G = nx.DiGraph()
        for t in reversed(self.T):
            G.add_node(f"Target {t}", node_type="Target")
        for w in reversed(self.W):
            G.add_node(f"Weapon {w}", node_type="Target")

        for t, w in itertools.product(self.T, self.W):
            if solution_array[w, t] == 1:
                G.add_edge(f"Weapon {w}", f"Target {t}")
        pos = nx.bipartite_layout(G, [f"Weapon {w}" for w in reversed(self.W)])

        nx.draw(
            G,
            pos=pos,
            nodelist=[f"Weapon {w}" for w in self.W],
            node_color="tab:blue",
            node_shape="o",
        )
        nx.draw(
            G,
            pos=pos,
            nodelist=[f"Target {t}" for t in self.T],
            node_color="tab:red",
            node_shape="X",
        )
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_labels(
            G,
            pos={f"Weapon {w}": pos[f"Weapon {w}"] - np.array([0.075, 0]) for w in self.W},
            horizontalalignment="right",
            labels={f"Weapon {w}": f"Weapon {w}" for w in self.W},
        )
        nx.draw_networkx_labels(
            G,
            pos={f"Target {t}": pos[f"Target {t}"] + np.array([0.075, 0]) for t in self.T},
            horizontalalignment="left",
            labels={f"Target {t}": f"Target {t}" for t in self.T},
        )
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.title(f"Expected surviving value post-engagement: {self.cost(solution_array):0.2f}")

    def cost(self, x):
        return sum(
            self.asset_values[k]
            * (
                1
                - sum(
                    self.asset_destruction_probs[j, k]
                    * (1 - sum(x[i, j] * self.target_destruction_probs[i, j] for i in self.W))
                    for j in self.T
                )
            )
            for k in self.K
        )

    def is_feasible(self, x):
        return np.all(x.sum(axis=1) == self.weapon_amunition_limits)

    def gamma_layer(self, gamma: float) -> Sequence[cirq.Operation]:
        for i in range(self.num_vars):
            for j in range(self.num_vars):
                if i != j and self.J_matrix[i, j] != 0:
                    yield cirq.ZZPowGate(exponent=-gamma * self.J_matrix[i, j] / np.pi)(
                        self.qubits[i], self.qubits[j]
                    )
            if self.h_vec[i] != 0:
                yield cirq.Rz(rads=2 * gamma * self.h_vec[i])(self.qubits[i])

    def beta_layer(self, beta: float) -> Sequence[cirq.Operation]:
        for qubit in self.qubits:
            yield cirq.Rx(rads=2 * beta)(qubit)

    def build_circuit(self, num_reps=2, store=False):
        gamma = sympy.symarray("𝛄", num_reps)
        beta = sympy.symarray("β", num_reps)

        # Start in the H|0> state.
        qaoa = cirq.Circuit(cirq.H.on_each(self.qubits))
        for k in range(num_reps):
            # Implement the U(gamma, C) operator.
            qaoa.append(self.gamma_layer(gamma[k]))
            # Implement the U(beta, B) operator.
            qaoa.append(self.beta_layer(beta[k]), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        # Measure
        qaoa += cirq.measure(*self.qubits)

        if store:
            self.qaoa_circuit = qaoa

        return qaoa

    def run_circuit(self, arr, reps=1_000):
        num_reps = len(arr) // 2
        gamma = sympy.symarray("𝛄", num_reps)
        beta = sympy.symarray("β", num_reps)
        gamma_val, beta_val = arr.reshape(2, num_reps)
        params = cirq.ParamResolver({**dict(zip(gamma, gamma_val)), **dict(zip(beta, beta_val))})
        x = self.sim.run(self.qaoa_circuit, param_resolver=params, repetitions=reps)
        return np.squeeze(list(x.records.values())[0])

    def energy(self, arr, shots=2):
        bit_strings = self.run_circuit(arr, reps=shots)
        spins = 2 * bit_strings - 1
        energy = self.offset + np.diag(spins @ self.J_matrix @ spins.T) + self.h_vec @ spins.T

        unique_energies, counts = np.unique(energy, axis=0, return_counts=True)

        p_min_energy = counts[unique_energies.argmin()]

        return -p_min_energy * np.abs(unique_energies.min())

    def optimal_bitstring(self, arr):
        bit_strings = self.run_circuit(arr, reps=50_000)

        # Find the solution that occurs the most often
        unique_bitstrings, counts = np.unique(bit_strings, axis=0, return_counts=True)

        most_probable = np.argmax(counts)
        return unique_bitstrings[most_probable][0 : self.num_targets * self.num_weapons].reshape(
            self.num_weapons, self.num_targets
        )

    def callback(self, xk):
        bitstrings = self.run_circuit(xk, reps=10_000)
        unique_bitstrings, counts = np.unique(bitstrings, axis=0, return_counts=True)
        unique_bitstrings = unique_bitstrings[counts.argsort()[::-1]]  # Sort by counts
        string_cost = np.array(
            [
                self.cost(
                    b[0 : self.num_weapons * self.num_targets].reshape(
                        self.num_weapons, self.num_targets
                    )
                )
                for b in unique_bitstrings
            ]
        )
        feasible = np.array(
            [
                self.is_feasible(
                    b[0 : self.num_weapons * self.num_targets].reshape(
                        self.num_weapons, self.num_targets
                    )
                )
                for b in unique_bitstrings
            ]
        )
        self._results.append(
            {
                "unique_bitstrings": unique_bitstrings,
                "costs": string_cost,
                "feasible": feasible,
                "counts": np.sort(counts)[::-1],
            }
        )
        self.plot_progress()

    def plot_progress(self):
        if len(self._results) <= 5:
            return
        clear_output(wait=True)
        xx = np.arange(len(self._results))
        yy = np.array([r["costs"][0] for r in self._results])
        ff = np.array(["tab:red" if not r["feasible"][0] else "tab:blue" for r in self._results])
        plt.xlabel("Optimizer Iteration")
        plt.ylabel("Objective value")
        plt.title("Progress")

        plt.scatter(xx, yy, color=ff)

        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.axhline(self.cost(self.exact_solution), linestyle="--", alpha=0.75, color="grey")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(
            handles=[
                mpatches.Patch(color="tab:red", label="Infeasible"),
                mpatches.Patch(color="tab:blue", label="Feasible"),
                mpatches.Patch(linestyle="--", color="grey", label="Optimal value"),
            ],
            bbox_to_anchor=(1, 0.5),
            loc="center left",
        )
        plt.show()

    def solve_qaoa(
        self,
        nreps=8,
        x0=None,
        method="COBYQA",
        shots=0,
        tol=1e-5,
        maxiter=None,
        jacobian=None,
        hessian=None,
        **kwargs,
    ):

        if method in [
            'Newton-CG', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'
        ]:
            jacobian, hessian = JACOBIAN, HESSIAN

        self.build_circuit(num_reps=nreps, store=True)

        assert shots > 1, "number of circuit executions must exceed a single shot."
        energy = lambda x: self.energy(x, shots=shots)

        self._results = []
        if x0 is None:
            x0 = np.ones(2 * nreps)
        res = scipy.optimize.minimize(
            energy,
            x0=x0,
            method=method,
            tol=tol,
            options={"maxiter": maxiter},
            callback=self.callback,
            jac=jacobian,
            hess=hessian,
            **kwargs,
        )
        return res
