import cirq
from cirq.circuits import InsertStrategy
import sympy

# index 15 is our magic state
qubits = cirq.LineQubit.range(16)

cir = cirq.Circuit()

# reset qubits
for q in qubits:
    cir.append([cirq.R(q)])

# set ket plus using hadamard
cir.append([cirq.H(qubits[0]),
            cirq.H(qubits[1]),
            cirq.H(qubits[3]),
            cirq.H(qubits[7]),
            cirq.H(qubits[15])], 
        strategy=InsertStrategy.NEW_THEN_INLINE)

# control qubits are @
# target qubits are X 
cir.append([cirq.CNOT(qubits[15], qubits[14]),
            cirq.CNOT(qubits[7], qubits[8]),
            cirq.CNOT(qubits[7], qubits[9]),
            cirq.CNOT(qubits[7], qubits[10]),
            cirq.CNOT(qubits[7], qubits[11]),
            cirq.CNOT(qubits[7], qubits[12]),
            cirq.CNOT(qubits[7], qubits[13]),
            cirq.CNOT(qubits[7], qubits[14])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

cir.append([cirq.CNOT(qubits[3], qubits[4]),
            cirq.CNOT(qubits[3], qubits[5]),
            cirq.CNOT(qubits[3], qubits[6]),
            cirq.CNOT(qubits[3], qubits[11]),
            cirq.CNOT(qubits[3], qubits[12]),
            cirq.CNOT(qubits[3], qubits[13]),
            cirq.CNOT(qubits[3], qubits[14])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

cir.append([cirq.CNOT(qubits[1], qubits[2]),
            cirq.CNOT(qubits[1], qubits[5]),
            cirq.CNOT(qubits[1], qubits[6]),
            cirq.CNOT(qubits[1], qubits[9]),
            cirq.CNOT(qubits[1], qubits[10]),
            cirq.CNOT(qubits[1], qubits[13]),
            cirq.CNOT(qubits[1], qubits[14])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

cir.append([cirq.CNOT(qubits[0], qubits[2]),
            cirq.CNOT(qubits[0], qubits[4]),
            cirq.CNOT(qubits[0], qubits[6]),
            cirq.CNOT(qubits[0], qubits[8]),
            cirq.CNOT(qubits[0], qubits[10]),
            cirq.CNOT(qubits[0], qubits[12]),
            cirq.CNOT(qubits[0], qubits[14])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

cir.append([cirq.CNOT(qubits[14], qubits[2]),
            cirq.CNOT(qubits[14], qubits[4]),
            cirq.CNOT(qubits[14], qubits[5]),
            cirq.CNOT(qubits[14], qubits[8]),
            cirq.CNOT(qubits[14], qubits[9]),
            cirq.CNOT(qubits[14], qubits[11])],
        strategy=InsertStrategy.NEW_THEN_INLINE)

cir.append([cirq.inverse(cirq.T(qubits[0])),
            cirq.inverse(cirq.T(qubits[1])),
            cirq.inverse(cirq.T(qubits[2])),
            cirq.inverse(cirq.T(qubits[3])),
            cirq.inverse(cirq.T(qubits[4])),
            cirq.inverse(cirq.T(qubits[5])),
            cirq.inverse(cirq.T(qubits[6])),
            cirq.inverse(cirq.T(qubits[7])),
            cirq.inverse(cirq.T(qubits[8])),
            cirq.inverse(cirq.T(qubits[9])),
            cirq.inverse(cirq.T(qubits[10])),
            cirq.inverse(cirq.T(qubits[11])),
            cirq.inverse(cirq.T(qubits[12])),
            cirq.inverse(cirq.T(qubits[13])),
            cirq.inverse(cirq.T(qubits[14])),],
        strategy=InsertStrategy.NEW_THEN_INLINE)

cir.append([cirq.H(qubits[0]),
            cirq.H(qubits[1]),
            cirq.H(qubits[2]),
            cirq.H(qubits[3]),
            cirq.H(qubits[4]),
            cirq.H(qubits[5]),
            cirq.H(qubits[6]),
            cirq.H(qubits[7]),
            cirq.H(qubits[8]),
            cirq.H(qubits[9]),
            cirq.H(qubits[10]),
            cirq.H(qubits[11]),
            cirq.H(qubits[12]),
            cirq.H(qubits[13]),
            cirq.H(qubits[14]),
            cirq.measure(qubits[0],  key='m1'),
            cirq.measure(qubits[1],  key='m2'),
            cirq.measure(qubits[2],  key='m3'),
            cirq.measure(qubits[3],  key='m4'),
            cirq.measure(qubits[4],  key='m5'),
            cirq.measure(qubits[5],  key='m6'),
            cirq.measure(qubits[6],  key='m7'),
            cirq.measure(qubits[7],  key='m8'),
            cirq.measure(qubits[8],  key='m9'),
            cirq.measure(qubits[9],  key='m10'),
            cirq.measure(qubits[10], key='m11'),
            cirq.measure(qubits[11], key='m12'),
            cirq.measure(qubits[12], key='m13'),
            cirq.measure(qubits[13], key='m14'),
            cirq.measure(qubits[14], key='m15'),],  # no need to measure index 15, that is our magic state
        strategy=InsertStrategy.NEW_THEN_INLINE)

print("CIRCUIT ====================")
print(cir)


# index 15=magic
m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15 = sympy.symbols('m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15')
# all those parities must be 0 for it to be even ( output of xor is 0 for even )
# so stop when true == ~0 & ~0 & ~0 & ~0 == ~(0 | 0 | 0 | 0)
# use demorgans
# https://arxiv.org/ftp/arxiv/papers/1208/1208.0928.pdf page 38 figure 33
evenParity = sympy.Not(sympy.Xor(m4,m5,m6,m7,m8,m9,m10,m11) | 
                       sympy.Xor(m1,m2,m3,m4,m5,m6,m7,m15) | 
                       sympy.Xor(m2,m3,m4,m5,m10,m11,m12,m13) | 
                       sympy.Xor(m1,m2,m5,m6,m9,m10,m13,m14))
# if special parity of all qubits is ODD, do Z 
specialParity = sympy.Xor(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15)

sim = cirq.Simulator()

# repeating until matched parities are all even
magicStateCir = cirq.Circuit(
        cirq.CircuitOperation(
            circuit=cir.freeze(), 
            use_repetition_ids=False,
            repeat_until=cirq.SympyCondition(evenParity)
        )
        # put corrections
    )

# magic state "correction"
# is odd, so do the Z
magicStateCir.append([cirq.Z(qubits[15]).with_classical_controls(specialParity)])

print(magicStateCir)

results = sim.simulate(magicStateCir)
print("RESULTS ====================")
print(results, end='\n\n')
print("FINAL STATE VECTOR ====================")
stateVector = results.final_state_vector
# print(stateVector)
print(cirq.dirac_notation(stateVector))


print("\n\nSIMULATING ====================")


fMap = {}
even_special_count = 0
odd_special_count = 0

# looking for 0.71|0⟩ + (0.5+0.5j)|1⟩
for i in range(1000):
    simResults = sim.simulate(magicStateCir)
    targetTensorResults = cirq.dirac_notation(simResults.get_state_containing_qubit(cirq.q(15)).target_tensor)
    m_o_q = "" # measurements_of_qubits 
    # print(simResults.measurements)
    for m in simResults.measurements:
        m_o_q += str(simResults.measurements[m][0])
    # print(m_o_q + "\n")

    if (sympy.Xor(m_o_q[0],m_o_q[1],m_o_q[2],m_o_q[3],m_o_q[4],m_o_q[5],m_o_q[6],m_o_q[7],m_o_q[8],m_o_q[9],m_o_q[10],m_o_q[11],m_o_q[12],m_o_q[13],m_o_q[14])): # is odd
        odd_special_count += 1
    else: # is even
        even_special_count += 1

    if (targetTensorResults) in fMap:
        fMap[targetTensorResults] += 1
    else:
        fMap[targetTensorResults] = 1
    # print(targetTensorResults)

print("CHECKING PARITY OF ALL QUBITS ====================")
print("EVEN SPECIAL PARITY:\t" + str(even_special_count)) # should give us 0.71|0⟩ + (0.5+0.5j)|1⟩
print("ODD SPECIAL PARITY:\t" + str(odd_special_count)) # should give us 0.71|0⟩ + (-0.5-0.5j)|1⟩


print("\nFINAL STATE VECTORS FREQUENCIES ====================")
for f in fMap:
    print(str(f) + "\t:\t" + str(fMap[f]))


print("\nDENSITY MATRIX (density_matrix_from_state_vector from cirq) ====================")
densityMatrix = cirq.density_matrix_from_state_vector(stateVector, indices=[15])
print(densityMatrix)

print("\nSINGLE QUBIT TOMOGRAPHY ====================")
tomography_result = cirq.experiments.single_qubit_state_tomography(sim, qubits[15], magicStateCir, 10000) 
print(tomography_result.data, end="\n\n\n")

# from IPython.display import FileLink
# cirq.to_json(magicStateCir, file_or_fn='MSD_15_to_1.json')
# FileLink('MSD_15_to_1.json')