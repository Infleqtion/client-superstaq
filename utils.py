from typing import Union
import os
import qiskit
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np



n_shots = 1000

def visualize_bloch_sphere(qc):
    sv = qiskit.quantum_info.Statevector(qc) 
    display(qiskit.visualization.plot_bloch_multivector(sv))


def swap_test(qc1, qc2):
    
    qc = qiskit.QuantumCircuit(3, 1)
    qc = qc.compose(qc1, [1])
    qc = qc.compose(qc2, [2])
    qc.barrier()
    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)
    qc.measure(0, 0)
 
    return qc


def plot_heatmap(circuits):
    simulator = qiskit.BasicAer.get_backend('qasm_simulator')
    
    l = len(circuits)
    hmap = np.eye(l)

    for i in range(l-1):
        for j in range(i+1, l):
            qc = swap_test(circuits[i], circuits[j])
            tqc = qiskit.transpile(qc, simulator)
            counts = simulator.run(tqc, shots=n_shots).result().get_counts()
            
            fid = (2*counts['0']/n_shots)-1
            hmap[i, j] = fid
            hmap[j, i] = fid
    print(hmap)
    plt.imshow(hmap)
    plt.colorbar()
    plt.show()


def show_bobs_qubit(qc):
    simulator = qiskit.BasicAer.get_backend('statevector_simulator')
    circ = qiskit.transpile(qc, simulator)
    sv = simulator.run(circ).result().get_statevector()

    bobs = sv[sv!=0]

    display(qiskit.visualization.plot_bloch_multivector(bobs))

    
def draw_circuits(circuits, figsize_scale=2.5, title="circuits"):
    nrows = np.floor(np.sqrt(len(circuits))).astype(int) 
    ncols = np.ceil(len(circuits) / nrows).astype(int)
    figure, axes_array = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(figsize_scale * ncols, figsize_scale * nrows),
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    figure.suptitle(title)
    # figure.patch.set_linewidth(10)
    # figure.patch.set_edgecolor("k")
    if nrows * ncols > 1:
        axes_array = axes_array.flatten()
    else:
        axes_array = [axes_array]
    for circuit, axes in zip(circuits, axes_array):
        axes.axis("off")
        circuit.draw("mpl", style={"name": "clifford"}, ax=axes)


def get_swap_test():
    """Get a Swap Test circuit.

    Measuring the ancilla qubit returns 0 with 100% probability if the states
    are identical, and 0-1 with 50-50% probability if the states are orthogonal.
    """
    qc = qiskit.QuantumCircuit(3, 1)
    qc.h(0)
    qc.cswap(0, 1, 2)
    qc.h(0)
    qc.measure([0], [0])
    return qc


def plot_fidelities_heatmap(fidelities):
    plt.figure(dpi=150)
    plt.imshow(fidelities)
    plt.colorbar()
