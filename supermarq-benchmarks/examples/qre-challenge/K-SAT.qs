namespace KSAT {
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Convert;

    /// ----------- Helper Function -----------

    /// # Summary
    /// Converts an array of boolean values into a string representation.
    /// # Input
    /// ## boolArray (Bool[]) : Array of boolean values
    ///
    /// # Output
    /// ## result (String) : String representation of the boolean array, where 
    /// `true` is represented by "1" and `false` is represented by "0"
    function BoolArrayAsString(boolArray : Bool[]) : String {
        mutable result = "";
        for b in boolArray {
            set result += (b ? "1" | "0");
        }
        return result;
    }

    /// ----------- Grover Oracle Definitions -----------

    /// # Summary
    /// Grover's oracle for OR operation.
    /// # Input
    /// ## register (Qubit[]) : Array of qubits
    ///
    /// ## target (Qubit) : Target qubit
    ///
    /// # Output
    /// ## None
    operation GroverOracleOR (register : Qubit[], target : Qubit) : Unit is Adj {
        X(target);
        H(target);
        OracleOr(register, target);
        H(target);
    }

    /// # Summary
    /// Grover's oracle for XOR operation.
    /// # Input
    /// ## register (Qubit[]) : Array of qubits
    ///
    /// ## target (Qubit) : Target qubit
    ///
    /// # Output
    /// ## None
    operation GroverOracleXOR (register : Qubit[], target : Qubit) : Unit is Adj {
        X(target);
        H(target);
        OracleXor(register, target);
        H(target);
    }

    /// ----------- Basic SAT Oracles -----------

    /// # Summary
    /// Implements an oracle for the OR operation on a quantum register.
    /// # Input
    /// ## queryRegister (Qubit[]) : Array of qubits used as the query register
    ///
    /// ## target (Qubit) : Target qubit
    ///
    /// # Output
    /// ## None
    operation OracleOr(queryRegister : Qubit[], target : Qubit) : Unit is Adj {
        for qubit in queryRegister {
            CNOT(qubit, target);
        }
        if (Length(queryRegister) > 1) {
            within {
                ApplyToEachA(X, queryRegister);
            } apply {
                Controlled X(queryRegister, target);
            }
        }
    }
    
    /// # Summary
    /// Implements an oracle for the XOR operation on a quantum register.
    /// # Input
    /// ## queryRegister (Qubit[]) : Array of qubits used as the query register
    ///
    /// ## target (Qubit) : Target qubit
    ///
    /// # Output
    /// ## None
    operation OracleXor (queryRegister : Qubit[], target : Qubit) : Unit is Adj {
        ApplyToEachA(CNOT(_, target), queryRegister);
    }

    /// # Summary
    /// Extracts the qubits and their corresponding flip values from the query 
    /// register based on the clause.
    /// # Input
    /// ## queryRegister (Qubit[]) : Array of qubits used as the query register
    ///
    /// ## clause ((Int, Bool)[]) : Array of tuples representing the index in 
    /// the query register and a boolean indicating whether to flip
    ///
    /// # Output
    /// ## (Qubit[], Bool[]) : A tuple containing the extracted qubits and their 
    /// flip values
    function GetClauseQubits (queryRegister : Qubit[], 
                              clause : (Int, Bool)[]) : (Qubit[], Bool[]) {
        mutable clauseQubits = [];
        mutable flip = [];
        for (index, isTrue) in clause {
            if (index >= Length(queryRegister)) {
                fail $"Index {index} out of range for queryRegister 
                length {Length(queryRegister)}";
            }
            set clauseQubits += [queryRegister[index]];
            set flip += [not isTrue];
        }
        return (clauseQubits, flip);
    }

    /// # Summary
    /// Implements an oracle for a SAT clause using a given query register and 
    /// target qubit.
    /// # Input
    /// ## queryRegister (Qubit[]) : Array of qubits used as the query register
    ///
    /// ## target (Qubit) : Target qubit
    ///
    /// ## clause ((Int, Bool)[]) : Array of tuples representing the index in 
    /// the query register and a boolean indicating whether to flip
    ///
    /// # Output
    /// ## None
    operation OracleSATClause (queryRegister : Qubit[], target : Qubit, 
                                clause : (Int, Bool)[]) : Unit is Adj {
        let (clauseQubits, flip) = GetClauseQubits(queryRegister, clause);
        within {
            ApplyPauliFromBitString(PauliX, true, flip, clauseQubits);
        } apply {
            OracleOr(clauseQubits, target);
        }
    }

    /// # Summary
    /// Evaluates OR clauses on a given query register and ancilla register 
    /// using a provided clause oracle.
    /// # Input
    /// ## queryRegister (Qubit[]) : Array of qubits used as the query register
    ///
    /// ## ancillaRegister (Qubit[]) : Array of qubits used as the ancilla register
    ///
    /// ## problem ((Int, Bool)[][]) : Array of clauses, each clause being an 
    /// array of tuples representing the index in the query register and a 
    /// boolean indicating whether to flip
    ///
    /// ## clauseOracle ((Qubit[], Qubit, (Int, Bool)[]) => Unit is Adj) : Oracle 
    /// function to evaluate individual clauses
    ///
    /// # Output
    /// ## None
    operation EvaluateOrClauses (queryRegister : Qubit[], 
                                 ancillaRegister : Qubit[], 
                                 problem : (Int, Bool)[][], 
                                 clauseOracle : ((Qubit[], Qubit, (Int, Bool)[]) 
                                => Unit is Adj)) : Unit is Adj {
        for clauseIndex in 0..Length(problem)-1 {
            clauseOracle(queryRegister, ancillaRegister[clauseIndex], problem[clauseIndex]);
        }
    }

    /// # Summary
    /// Implements a SAT oracle using a given query register and target qubit, 
    /// evaluating multiple OR clauses.
    /// # Input
    /// ## queryRegister (Qubit[]) : Array of qubits used as the query register
    ///
    /// ## target (Qubit) : Target qubit
    ///
    /// ## problem ((Int, Bool)[][]) : Array of clauses, each clause being an 
    /// array of tuples representing the index in the query register and a 
    /// boolean indicating whether to flip
    ///
    /// # Output
    /// ## None
    operation OracleSAT (queryRegister : Qubit[], target : Qubit, 
                          problem : (Int, Bool)[][]) : Unit is Adj {
        use ancillaRegister = Qubit[Length(problem)];
        within {
            EvaluateOrClauses(queryRegister, ancillaRegister, problem, OracleSATClause);
        } apply {
            Controlled X(ancillaRegister, target);
        }
    }

    /// ----------- Grover's Algorithm Implementation -----------

    /// # Summary
    /// Implements an oracle converter that applies a marking oracle and 
    /// increments the number of oracle calls.
    /// # Input
    /// ## markingOracle ((Qubit[], Qubit) => Unit is Adj) : Oracle function 
    /// for marking
    ///
    /// ## register (Qubit[]) : Array of qubits used as the register
    ///
    /// ## numOracleCalls (Int) : Number of times the oracle has been called
    ///
    /// # Output
    /// ## None
    operation OracleConverterImpl(markingOracle : ((Qubit[], Qubit) => Unit is Adj), 
                                  register : Qubit[], 
                                  numOracleCalls : Int) : Unit {
        use target = Qubit();
        X(target);
        H(target);
        markingOracle(register, target);
        let numOracleCalls = numOracleCalls + 1;
        H(target);
        let measuredResult = M(target);
        if (measuredResult == One) {
            X(target);
        }
    }

    /// # Summary
    /// Returns an oracle converter function that can be applied to a qubit 
    /// register.
    /// # Input
    /// ## markingOracle ((Qubit[], Qubit) => Unit is Adj) : Oracle function 
    /// for marking
    ///
    /// ## numOracleCalls (Int) : Number of times the oracle has been called
    ///
    /// # Output
    /// ## (Qubit[] => Unit) : Function that applies the oracle converter to a 
    /// qubit register
    function OracleConverter (markingOracle : ((Qubit[], Qubit) => Unit is Adj), 
                              numOracleCalls : Int) : (Qubit[] => Unit) {
        return OracleConverterImpl(markingOracle, _, numOracleCalls);
    }

    /// # Summary
    /// Implements Grover's algorithm loop using a given oracle and a specified 
    /// number of iterations.
    /// # Input
    /// ## register (Qubit[]) : Array of qubits used as the register
    ///
    /// ## oracle ((Qubit[], Qubit) => Unit is Adj) : Oracle function for 
    /// Grover's algorithm
    ///
    /// ## iterations (Int) : Number of iterations to perform
    ///
    /// ## numOracleCalls (Int) : Number of times the oracle has been called
    ///
    /// # Output
    /// ## None
    operation GroversAlgorithmLoop (register : Qubit[], oracle : ((Qubit[], Qubit) 
    => Unit is Adj), iterations : Int, numOracleCalls : Int) : Unit {
        let phaseOracle = OracleConverter(oracle, numOracleCalls);
        ApplyToEach(H, register);
        for i in 1 .. iterations {
            phaseOracle(register);
            within {
                ApplyToEachA(H, register);
                ApplyToEachA(X, register);
            } apply {
                Controlled Z(Most(register), Tail(register));
            }
        }
    }

    /// # Summary
    /// Measures each qubit in the provided array and returns the results.
    /// # Input
    /// ## targets (Qubit[]) : Array of qubits to be measured
    ///
    /// # Output
    /// ## Result[] : Array of measurement results
    operation MultiM (targets : Qubit[]) : Result[] {
        return ForEach(M, targets);
    }

    /// # Summary
    /// Implements a universal version of Grover's algorithm for a given problem 
    /// size and oracle function.
    /// # Input
    /// ## N (Int) : Number of qubits in the register
    ///
    /// ## oracle ((Qubit[], Qubit) => Unit is Adj) : Oracle function for 
    /// Grover's algorithm
    ///
    /// # Output
    /// ## (Bool[], Int) : Tuple containing the solution as a boolean array and 
    /// the number of oracle calls made
    operation UniversalGroversAlgorithm (N : Int, oracle : ((Qubit[], Qubit) 
    => Unit is Adj)) : (Bool[], Int) {
        mutable answer = [false, size = N];
        use (register, output) = (Qubit[N], Qubit());
        mutable correct = false;
        mutable iter = 1;
        mutable numOracleCalls = 0;
        repeat {
            GroversAlgorithmLoop(register, oracle, iter, numOracleCalls);
            let res = MultiM(register);
            oracle(register, output);
            set numOracleCalls += 1;
            if MResetZ(output) == One {
                set correct = true;
                set answer = ResultArrayAsBoolArray(res);
            }
            ResetAll(register);
        } until (correct or iter > 100)
        fixup {
            set iter *= 2;
        }
        if not correct {
            fail "Failed to find an answer";
        }
        return (answer, numOracleCalls);
    }

    /// ----------- k-SAT Tests -----------

    /// # Summary
    /// Estimates the number of iterations required for Grover's algorithm based 
    /// on the number of variables.
    /// # Input
    /// ## numVariables (Int) : Number of variables in the SAT problem
    ///
    /// # Output
    /// ## Int : Estimated number of Grover iterations
    function EstimateGroverIterations(numVariables : Int) : Int {
        return 2^(numVariables / 2);
    }

    /// # Summary
    /// Solves a SAT problem with a given number of variables and clauses using 
    /// Grover's algorithm.
    /// # Input
    /// ## numVariables (Int) : Number of variables in the SAT problem
    ///
    /// ## clauses ((Int, Bool)[][]) : Array of clauses, each clause being an 
    /// array of tuples representing the index in the query register and a 
    /// boolean indicating whether to flip
    ///
    /// # Output
    /// ## None
    operation SolveSAT(numVariables : Int, clauses : (Int, Bool)[][]) : Unit {
        let iterations = EstimateGroverIterations(numVariables);
        Message($"Solving SAT problem with {numVariables} variables and 
        {Length(clauses)} clauses using {iterations} Grover iterations.");
        let (solution, numOracleCalls) = UniversalGroversAlgorithm(numVariables, 
                                                    OracleSAT(_, _, clauses));
        Message($"Solution: {BoolArrayAsString(solution)}");
        Message($"Number of oracle calls: {numOracleCalls}");
    }

    /// # Summary
    /// Solves a SAT problem with 8 variables and a predefined set of clauses.
    /// # Input
    /// ## None
    ///
    /// # Output
    /// ## None
    @EntryPoint()
    operation SolveSAT8() : Unit {
        let clauses = [
            [(0, true), (1, false), (2, true)],
            [(0, false), (1, true), (2, false)],
            [(0, true), (3, true), (4, false)],
            [(1, false), (2, true), (3, true)],
            [(1, true), (3, false), (4, true)],
            [(2, true), (3, true), (4, false)]
        ];
        SolveSAT(8, clauses);
    }

    /// # Summary
    /// Solves a SAT problem with 16 variables and a predefined set of clauses.
    /// # Input
    /// ## None
    ///
    /// # Output
    /// ## None
    operation SolveSAT16() : Unit {
        let clauses = [
            [(0, true), (1, false), (2, true)],
            [(0, false), (1, true), (2, false)],
            [(0, true), (3, true), (4, false)],
            [(1, false), (2, true), (3, true)],
            [(1, true), (3, false), (4, true)],
            [(2, true), (3, true), (4, false)],
            [(4, true), (5, false), (6, true)],
            [(4, false), (5, true), (6, false)],
            [(0, true), (2, true), (5, true)],
            [(3, false), (1, true), (6, false)]
        ];
        SolveSAT(16, clauses);
    }

    /// # Summary
    /// Solves a SAT problem with 24 variables and a predefined set of clauses.
    /// # Input
    /// ## None
    ///
    /// # Output
    /// ## None
    operation SolveSAT24() : Unit {
        let clauses = [
            [(0, true), (1, false), (2, true)],
            [(0, false), (1, true), (2, false)],
            [(0, true), (3, true), (4, false)],
            [(1, false), (2, true), (3, true)],
            [(1, true), (3, false), (4, true)],
            [(2, true), (3, true), (4, false)],
            [(4, true), (5, false), (6, true)],
            [(4, false), (5, true), (6, false)],
            [(0, true), (2, true), (5, true)],
            [(3, false), (1, true), (6, false)],
            [(7, true), (8, false), (9, true)],
            [(7, false), (8, true), (9, false)]
        ];
        SolveSAT(24, clauses);
    }

    /// # Summary
    /// Solves a SAT problem with 32 variables and a predefined set of clauses.
    /// # Input
    /// ## None
    ///
    /// # Output
    /// ## None
    operation SolveSAT32() : Unit {
        let clauses = [
            [(0, true), (1, false), (2, true)],
            [(0, false), (1, true), (2, false)],
            [(0, true), (3, true), (4, false)],
            [(1, false), (2, true), (3, true)],
            [(1, true), (3, false), (4, true)],
            [(2, true), (3, true), (4, false)],
            [(4, true), (5, false), (6, true)],
            [(4, false), (5, true), (6, false)],
            [(0, true), (2, true), (5, true)],
            [(3, false), (1, true), (6, false)],
            [(7, true), (8, false), (9, true)],
            [(7, false), (8, true), (9, false)],
            [(6, true), (7, false), (8, true)],
            [(5, false), (6, true), (7, true)],
            [(4, true), (5, false), (6, true)],
            [(3, true), (4, false), (5, true)]
        ];
        SolveSAT(32, clauses);
    }

    /// # Summary
    /// Solves a SAT problem with 40 variables and a predefined set of clauses.
    /// # Input
    /// ## None
    ///
    /// # Output
    /// ## None
    operation SolveSAT40() : Unit {
        let clauses = [
            [(0, true), (1, false), (2, true)],
            [(0, false), (1, true), (2, false)],
            [(0, true), (3, true), (4, false)],
            [(1, false), (2, true), (3, true)],
            [(1, true), (3, false), (4, true)],
            [(2, true), (3, true), (4, false)],
            [(4, true), (5, false), (6, true)],
            [(4, false), (5, true), (6, false)],
            [(0, true), (2, true), (5, true)],
            [(3, false), (1, true), (6, false)],
            [(7, true), (8, false), (9, true)],
            [(7, false), (8, true), (9, false)],
            [(6, true), (7, false), (8, true)],
            [(5, false), (6, true), (7, true)],
            [(4, true), (5, false), (6, true)],
            [(3, true), (4, false), (5, true)],
            [(2, true), (3, false), (4, true)],
            [(1, true), (2, false), (3, true)]
        ];
        SolveSAT(40, clauses);
    }

    /// # Summary
    /// Solves a SAT problem with 48 variables and a predefined set of clauses.
    /// # Input
    /// ## None
    ///
    /// # Output
    /// ## None
    operation SolveSAT48() : Unit {
        let clauses = [
            [(0, true), (1, false), (2, true)],
            [(0, false), (1, true), (2, false)],
            [(0, true), (3, true), (4, false)],
            [(1, false), (2, true), (3, true)],
            [(1, true), (3, false), (4, true)],
            [(2, true), (3, true), (4, false)],
            [(4, true), (5, false), (6, true)],
            [(4, false), (5, true), (6, false)],
            [(0, true), (2, true), (5, true)],
            [(3, false), (1, true), (6, false)],
            [(7, true), (8, false), (9, true)],
            [(7, false), (8, true), (9, false)],
            [(6, true), (7, false), (8, true)],
            [(5, false), (6, true), (7, true)],
            [(4, true), (5, false), (6, true)],
            [(3, true), (4, false), (5, true)],
            [(2, true), (3, false), (4, true)],
            [(1, true), (2, false), (3, true)],
            [(0, true), (1, false), (2, true)],
            [(9, true), (8, false), (7, true)]
        ];
        SolveSAT(48, clauses);
    }

    /// # Summary
    /// Solves a SAT problem with 56 variables and a predefined set of clauses.
    /// # Input
    /// ## None
    ///
    /// # Output
    /// ## None
    operation SolveSAT56() : Unit {
        let clauses = [
            [(0, true), (1, false), (2, true)],
            [(0, false), (1, true), (2, false)],
            [(0, true), (3, true), (4, false)],
            [(1, false), (2, true), (3, true)],
            [(1, true), (3, false), (4, true)],
            [(2, true), (3, true), (4, false)],
            [(4, true), (5, false), (6, true)],
            [(4, false), (5, true), (6, false)],
            [(0, true), (2, true), (5, true)],
            [(3, false), (1, true), (6, false)],
            [(7, true), (8, false), (9, true)],
            [(7, false), (8, true), (9, false)],
            [(6, true), (7, false), (8, true)],
            [(5, false), (6, true), (7, true)],
            [(4, true), (5, false), (6, true)],
            [(3, true), (4, false), (5, true)],
            [(2, true), (3, false), (4, true)],
            [(1, true), (2, false), (3, true)],
            [(0, true), (1, false), (2, true)],
            [(9, true), (8, false), (7, true)],
            [(8, true), (7, false), (6, true)],
            [(7, true), (6, false), (5, true)]
        ];
        SolveSAT(56, clauses);
    }

    /// # Summary
    /// Solves a SAT problem with 64 variables and a predefined set of clauses.
    /// # Input
    /// ## None
    ///
    /// # Output
    /// ## None
    operation SolveSAT64() : Unit {
        let clauses = [
            [(0, true), (1, false), (2, true)],
            [(0, false), (1, true), (2, false)],
            [(0, true), (3, true), (4, false)],
            [(1, false), (2, true), (3, true)],
            [(1, true), (3, false), (4, true)],
            [(2, true), (3, true), (4, false)],
            [(4, true), (5, false), (6, true)],
            [(4, false), (5, true), (6, false)],
            [(0, true), (2, true), (5, true)],
            [(3, false), (1, true), (6, false)],
            [(7, true), (8, false), (9, true)],
            [(7, false), (8, true), (9, false)],
            [(6, true), (7, false), (8, true)],
            [(5, false), (6, true), (7, true)],
            [(4, true), (5, false), (6, true)],
            [(3, true), (4, false), (5, true)],
            [(2, true), (3, false), (4, true)],
            [(1, true), (2, false), (3, true)],
            [(0, true), (1, false), (2, true)],
            [(9, true), (8, false), (7, true)],
            [(8, true), (7, false), (6, true)],
            [(7, true), (6, false), (5, true)],
            [(6, true), (5, false), (4, true)],
            [(5, true), (4, false), (3, true)]
        ];
        SolveSAT(64, clauses);
    }
}
