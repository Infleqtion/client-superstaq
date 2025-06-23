import cirq
from cirq.circuits import InsertStrategy
import sympy

# index 0 is our magic state
qubits = cirq.LineQubit.range(8)

cir = cirq.Circuit()

# reset qubits
for q in qubits:
    cir.append([cirq.R(q)])

# set ket plus using hadamard
cir.append([cirq.H(qubits[0]),
            cirq.H(qubits[1]),
            cirq.H(qubits[2]),
            cirq.H(qubits[4])], 
        strategy=InsertStrategy.NEW_THEN_INLINE)

# control qubits are @
# target qubits are X 
cir.append([cirq.CNOT(qubits[0], qubits[3])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

cir.append([cirq.CNOT(qubits[3], qubits[5]),
            cirq.CNOT(qubits[3], qubits[6])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

cir.append([cirq.CNOT(qubits[4], qubits[5]),
            cirq.CNOT(qubits[4], qubits[6]),
            cirq.CNOT(qubits[4], qubits[7])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

cir.append([cirq.CNOT(qubits[2], qubits[3]),
            cirq.CNOT(qubits[2], qubits[6]),
            cirq.CNOT(qubits[2], qubits[7])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

cir.append([cirq.CNOT(qubits[1], qubits[3]),
            cirq.CNOT(qubits[1], qubits[5]),
            cirq.CNOT(qubits[1], qubits[7])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

# adding S gate
cir.append([cirq.S(qubits[1]),
            cirq.S(qubits[2]),
            cirq.S(qubits[3]),
            cirq.S(qubits[4]),
            cirq.S(qubits[5]),
            cirq.S(qubits[6]),
            cirq.S(qubits[7])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

# no need to measure index 0, that is our magic state
cir.append([cirq.H(qubits[1]),
            cirq.H(qubits[2]),
            cirq.H(qubits[3]),
            cirq.H(qubits[4]),
            cirq.H(qubits[5]),
            cirq.H(qubits[6]),
            cirq.H(qubits[7]),
            cirq.measure(qubits[1], key='m6'),
            cirq.measure(qubits[2], key='m5'),
            cirq.measure(qubits[3], key='m4'),
            cirq.measure(qubits[4], key='m3'),
            cirq.measure(qubits[5], key='m2'),
            cirq.measure(qubits[6], key='m1'),
            cirq.measure(qubits[7], key='m0'),], 
        strategy=InsertStrategy.NEW_THEN_INLINE)

print("CIRCUIT ====================")
print(cir)


# index 0=magic
m0, m1, m2, m3, m4, m5, m6 = sympy.symbols('m0 m1 m2 m3 m4 m5 m6')
# all those parities must be 0 for it to be even ( so output of xor is 0 for even )
# so stop when true == ~0 & ~0 & ~0 == ~(0 | 0 | 0)
# use demorgans
# https://arxiv.org/pdf/0803.0272 page 12 figure 19
evenParity = sympy.Not(sympy.Xor(m0,m1,m2,m3) | sympy.Xor(m0,m1,m4,m5) | sympy.Xor(m0,m2,m4,m6))
# if special parity is EVEN, do Z 
specialParity = sympy.Xor(m4,m5,m6)

sim = cirq.Simulator()

# repeating until matched parities
magicStateCir = cirq.Circuit(
        cirq.CircuitOperation(
            circuit=cir.freeze(), 
            use_repetition_ids=False,
            repeat_until=cirq.SympyCondition(evenParity)
        )
    )

# magic state "correction"
# is even, so do the Z
magicStateCir.append([cirq.Z(qubits[0]).with_classical_controls(sympy.Not(specialParity))])

print(magicStateCir)

results = sim.simulate(magicStateCir)
print("\nRESULTS ====================")
print(results, end='\n\n')
print("\nFINAL STATE VECTOR =========")
stateVector = results.final_state_vector
# print(stateVector)
print(cirq.dirac_notation(stateVector))


print("\n\nSIMULATING ====================")


# looking for 0.71|0⟩ + (0.71j)|1⟩
fMap = {}
for i in range(1000):
    targetTensorResults = cirq.dirac_notation(sim.simulate(magicStateCir).get_state_containing_qubit(cirq.q(0)).target_tensor)
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
tomography_result = cirq.experiments.single_qubit_state_tomography(sim, qubits[0], magicStateCir, 10000) 
print(tomography_result.data)

# from IPython.display import FileLink
# cirq.to_json(magicStateCir, file_or_fn='MSD_7_to_1.json')
# FileLink('MSD_7_to_1.json')