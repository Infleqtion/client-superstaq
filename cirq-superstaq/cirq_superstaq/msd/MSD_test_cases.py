# ruff: noqa: ERA001, T201

import cirq
import sympy
from cirq.circuits import InsertStrategy

from MSD_5_to_1.py import msd_5_to_1 
from MSD_15_to_1.py import msd_15_to_1 
from MSD_7_to_1.py import msd_7_to_1 


# TESTING 5-to-1 MSD
print("5 to 1 MAGIC STATE DISTILLATION ====================")
qubits_5 = cirq.LineQubit.range(5)
sim = cirq.Simulator()
magicStateCir = msd_5_to_1(qubits_5)
print(magicStateCir)

results = sim.simulate(magicStateCir)
print("\nRESULTS ====================")
print(results, end="\n\n")
print("\nFINAL STATE VECTOR =========")
stateVector = results.final_state_vector
# print(stateVector)
print(cirq.dirac_notation(stateVector))


print("\n\nSIMULATING ====================")


fMap: dict[str, int] = {}
for _i in range(1000):
    targetTensorResults = cirq.dirac_notation(
        sim.simulate(magicStateCir).get_state_containing_qubit(cirq.q(0)).target_tensor
    )
    if targetTensorResults in fMap:
        fMap[targetTensorResults] += 1
    else:
        fMap[targetTensorResults] = 1
    # print(targetTensorResults)

print("\nFINAL STATE VECTORS FREQUENCIES ====================")
for f in fMap:
    print(str(f) + "\t:\t" + str(fMap[f]))


print("\nDENSITY MATRIX (density_matrix_from_state_vector from cirq) ====================")
densityMatrix = cirq.density_matrix_from_state_vector(stateVector, indices=[0])
print(densityMatrix)

print("\nSINGLE QUBIT TOMOGRAPHY ====================")
tomo_res = cirq.experiments.single_qubit_state_tomography(sim, qubits_5[0], magicStateCir, 1000)
print(tomo_res.data)



# TESTING 15-to-1 MSD
print("15 to 1 MAGIC STATE DISTILLATION ====================")
qubits_15 = cirq.LineQubit.range(16)
sim = cirq.Simulator()
magicStateCir = msd_15_to_1(qubits_15)
print(magicStateCir)

results = sim.simulate(magicStateCir)
print("RESULTS ====================")
print(results, end="\n\n")
print("FINAL STATE VECTOR ====================")
stateVector = results.final_state_vector
# print(stateVector)
print(cirq.dirac_notation(stateVector))


print("\n\nSIMULATING ====================")


fMap: dict[str, int] = {}
even_special_count = 0
odd_special_count = 0

# looking for 0.71|0⟩ + (0.5+0.5j)|1⟩
for _i in range(1000):
    simResults = sim.simulate(magicStateCir)
    targetTensorResults = cirq.dirac_notation(
        simResults.get_state_containing_qubit(cirq.q(15)).target_tensor
    )
    m_o_q: str = ""  # measurements_of_qubits
    # print(simResults.measurements)
    for m in simResults.measurements:
        m_o_q += str(simResults.measurements[m][0])
    # print(m_o_q + "\n")

    if sympy.Xor(
        m_o_q[0],
        m_o_q[1],
        m_o_q[2],
        m_o_q[3],
        m_o_q[4],
        m_o_q[5],
        m_o_q[6],
        m_o_q[7],
        m_o_q[8],
        m_o_q[9],
        m_o_q[10],
        m_o_q[11],
        m_o_q[12],
        m_o_q[13],
        m_o_q[14],
    ):  # is odd
        odd_special_count += 1
    else:  # is even
        even_special_count += 1

    if (targetTensorResults) in fMap:
        fMap[targetTensorResults] += 1
    else:
        fMap[targetTensorResults] = 1
    # print(targetTensorResults)

print("CHECKING PARITY OF ALL QUBITS ====================")
print("EVEN SPECIAL PARITY:\t" + str(even_special_count))  # should give us 0.71|0⟩ + (0.5+0.5j)|1⟩
print("ODD SPECIAL PARITY:\t" + str(odd_special_count))  # should give us 0.71|0⟩ + (-0.5-0.5j)|1⟩

print("\nFINAL STATE VECTORS FREQUENCIES ====================")
for f in fMap:
    print(str(f) + "\t:\t" + str(fMap[f]))


print("\nDENSITY MATRIX (density_matrix_from_state_vector from cirq) ====================")
densityMatrix = cirq.density_matrix_from_state_vector(stateVector, indices=[15])
print(densityMatrix)

print("\nSINGLE QUBIT TOMOGRAPHY ====================")
tomo_res = cirq.experiments.single_qubit_state_tomography(sim, qubits_15[15], magicStateCir, 10000)
print(tomo_res.data, end="\n\n\n")



# TESTING 7-to-1 MSD
print("7 to 1 MAGIC STATE DISTILLATION ====================")
qubits_7 = cirq.LineQubit.range(8)
sim = cirq.Simulator()
magicStateCir = msd_7_to_1(qubits_7)
print(magicStateCir)

results = sim.simulate(magicStateCir)
print("\nRESULTS ====================")
print(results, end="\n\n")
print("\nFINAL STATE VECTOR =========")
stateVector = results.final_state_vector
# print(stateVector)
print(cirq.dirac_notation(stateVector))


print("\n\nSIMULATING ====================")


# looking for 0.71|0⟩ + (0.71j)|1⟩
fMap: dict[str, int] = {}
for _i in range(1000):
    targetTensorResults = cirq.dirac_notation(
        sim.simulate(magicStateCir).get_state_containing_qubit(cirq.q(0)).target_tensor
    )
    if targetTensorResults in fMap:
        fMap[targetTensorResults] += 1
    else:
        fMap[targetTensorResults] = 1
    # print(targetTensorResults)

print("\nFINAL STATE VECTORS FREQUENCIES ====================")
for f in fMap:
    print(str(f) + "\t:\t" + str(fMap[f]))


print("\nDENSITY MATRIX (density_matrix_from_state_vector from cirq) ====================")
densityMatrix = cirq.density_matrix_from_state_vector(stateVector, indices=[0])
print(densityMatrix)

print("\nSINGLE QUBIT TOMOGRAPHY ====================")
tomo_res = cirq.experiments.single_qubit_state_tomography(sim, qubits_7[0], magicStateCir, 10000)
print(tomo_res.data)