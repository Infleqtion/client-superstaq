# Architecture-Level Quantum Resource Estimator

## Spencer Dearman

### **Abstract**

This document outlines the design and development of an architecture-level Quantum Resource Estimation (QRE) tool. Unlike existing QRE tools, which are mainly focused on quantum algorithm-level estimations, this tool aims to address the complexities involved in quantum architecture.


### **Background and Motivation**

Existing QRE tools are limited in their ability to provide insights at the architecture level, particularly when comparing Surface Codes with LRESC. Current tools just highlight the inherent advantage of LRESC in saving qubits, a conclusion that can be reached without extensive QRE analysis. This project aims to build a more advanced QRE tool that considers other factors such as Movement, Atom Reloading, Code Teleportation, Connectivity, Degree of Parallelization, and other architectural assumptions, providing more meaningful and actionable insights.

 
### **Current Resource Estimation Limitations**

_Limitations of Azure Quantum Resource Estimator and Qualtran_

Both the Azure Quantum Resource Estimator (AQRE) and Qualtran are leading tools used in quantum resource estimation. However, despite their popularity, these tools are predominantly algorithm-centric. This narrow focus on algorithms presents a significant limitation when attempting to produce realistic estimations for specific and more complex quantum systems, as they largely disregard the hardware-specific characteristics that are crucial.

_Algorithmic Focus and Lack of Customizability_

The primary issue with AQRE and Qualtran lies in their purely algorithmic approach. These platforms are designed to optimize quantum algorithms, such as those dependent on magic state distillation (MSD), which is currently considered one of the most efficient methods. However, this focus on optimizing algorithms comes at the expense of considering the underlying hardware architecture, which is vital for producing accurate and realistic resource estimations. For example, the AQRE and Qualtran platforms lack the ability to account for specifics such as qubit movement, local device constraints, or the physical layout of qubits, which are critical factors in the architecture of quantum systems.

In our attempts to push AQRE to its limits, we quickly realized the extent of its limitations. We explored various aspects, including the potential for movement optimization and the use of iterative decoding, but found that the tool was simply not flexible enough to accommodate these considerations. For instance, the tool could not provide concrete estimations for systems like Trapped Ions and Neutral Atoms, where the architecture’s specific needs, such as the necessity of movement to save resources, are not easily captured by a tool that is hard-coded and not dynamic.

_Hard-Coded Architectures and Inflexibility_

One of the most significant challenges we encountered with these tools is their lack of flexibility. Quantum resource estimators like AQRE and BenchQ are inherently designed to optimize predefined algorithms, and they are not equipped to adjust to the dynamic needs of various quantum architectures. For example, when examining the Alice and Bob architecture, or when dealing with Low-Density Parity-Check (LDPC) codes, the limitations became apparent. The hard-coded nature of these tools meant that any deviation from the predefined architecture resulted in inefficiencies and suboptimal results. The lack of support for critical features such as movement and flip-chip architecture further highlighted the need for a more adaptable tool.

 
### **Qubit Architecture**

_Synthetic Qubits (Superconducting)_

Synthetic qubits, particularly those based on superconducting circuits, are one of the leading platforms in quantum computing. These qubits are realized using Josephson junctions, which allow for the creation of a two-level quantum system with low decoherence times and high controllability. The architecture for superconducting qubits is highly developed, with established techniques for qubit manipulation, readout, and error correction. However, the physical infrastructure required, such as dilution refrigerators, makes scalability and qubit connectivity challenging, especially when dealing with large numbers of qubits.

_Neutral Atom Simulations_

Neutral atom qubits, on the other hand, use individual atoms trapped in optical lattices or tweezers as qubits. These qubits are manipulated using laser pulses to perform quantum gates and entanglement operations. Neutral atom simulations are a promising avenue for large-scale quantum computing due to their natural scalability and the ability to arrange atoms in highly connected, flexible geometries. However, challenges remain in terms of maintaining coherence over long periods and achieving high-fidelity operations across large numbers of qubits.

_Comparison and Implications for QRE_

The choice of qubit architecture—synthetic versus neutral atom—has significant implications for QRE. Superconducting qubits often require careful consideration of connectivity due to their fixed physical layout, while neutral atoms offer more flexibility in qubit arrangement but face challenges in operation fidelity. A comprehensive QRE tool must account for these differences in architecture, adjusting resource estimations based on the type of qubit used, its coherence properties, and the specific operational constraints associated with each platform.

### **Requirements**

_Functional Requirements_

- **Movement Analysis:** The tool must evaluate the impact of qubit movement within the architecture.

- **Atom Reloading:** Consideration of the frequency and impact of atom reloading within quantum operations.

- **Code Teleportation:** Ability to model and assess the implications of code teleportation on qubit resources.

- **Connectivity:** Analysis of qubit connectivity and its effect on the overall architecture.

- **Degree of Parallelization:** Estimation of the impact of parallel operations on qubit usage and efficiency.

- **Custom Assumptions:** Flexibility to incorporate various architectural assumptions and parameters.

_Non-Functional Requirements_

- **Performance:** The tool should provide results in a reasonable timeframe, even for large-scale quantum architectures.

- **Scalability:** Ability to scale with increasing complexity of quantum systems.

- **Usability:** User-friendly interface for easy input of parameters and interpretation of results.

- **Extensibility:** Design should allow for future enhancements and the addition of new features.

### **System Architecture**

_High-Level Architecture_

- **Input Module:** Accepts user-defined parameters for the quantum architecture, including movement constraints, reloading rates, and connectivity graphs.

- **Analysis Engine:** Core computational module that performs the QRE calculations, considering all the defined parameters and assumptions.

- **Output Module:** Generates reports and visualizations, providing insights into qubit resource usage and architectural trade-offs.

- **User Interface:** A GUI that allows users to interact with the tool, input data, run simulations, and view results.

_Detailed Architecture_

- **Movement Analysis Subsystem:** Implements algorithms to estimate the impact of qubit movement.

- **Atom Reloading Subsystem:** Models the atom reloading process and its effect on qubit availability.

- **Code Teleportation Subsystem:** Evaluates teleportation strategies and their resource implications.

- **Connectivity Analysis Subsystem:** Analyzes qubit connectivity patterns and their impact on the architecture.

- **Parallelization Analysis Subsystem:** Assesses the degree of parallelization and its effect on qubit usage.

### **Conclusion**

This QRE tool aims to fill the gap in architecture-level quantum resource estimation, providing more precise and actionable insights than existing tools. With a focus on scalability, flexibility, and usability, this tool will be a valuable asset for researchers and engineers working in the field of quantum computing.
