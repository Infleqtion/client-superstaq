# Architecture-Level Quantum Resource Estimator

## Spencer Dearman

### **Abstract**

This document outlines the design and development of an architecture-level Quantum Resource Estimation (QRE) tool. Unlike existing QRE tools, which are mainly focused on quantum algorithm-level estimations, this tool aims to address the complexities involved in quantum architecture.


### **Background and Motivation**

Existing QRE tools are limited in their ability to provide insights at the architecture level, particularly when comparing Surface Codes with LRESC. Current tools just highlight the inherent advantage of LRESC in saving qubits, a conclusion that can be reached without extensive QRE analysis. This project aims to build a more advanced QRE tool that considers other factors such as Movement, Atom Reloading, Code Teleportation, Connectivity, Degree of Parallelization, and other architectural assumptions, providing more meaningful and actionable insights.

While there are existing software implementations that perform architecture-level resource estimations, a centralized, hardware-agnostic tool is needed to consolidate these efforts. Some examples of architecture-level estimations can be found in the following papers:
- [How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits](https://scirate.com/arxiv/1905.09749)
- [Building a fault-tolerant quantum computer using concatenated cat codes](https://scirate.com/arxiv/2012.04108)
- [LDPC-cat codes for low-overhead quantum computing in 2D](https://scirate.com/arxiv/2401.09541)
- [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://scirate.com/arxiv/2306.08585)

 
### **Current Resource Estimation Limitations**

_Limitations of Current Quantum Resource Estimators_

The Azure Quantum Resource Estimator (AQRE), Qualtran, BenchQ, and pyLIQTR are prominent tools in quantum resource estimation. Despite their widespread use, these tools are primarily algorithm-centric, with limited flexibility. By default, they assume a surface quantum error correction code with lattice surgery and magic state distillation, offering no customization of these parameters. In our research, we identified alternative QEC configurations that diverge from MSD and lattice surgery; however, testing these configurations was not feasible due to the tools’ rigid, hard-coded options.

This algorithmic focus imposes a significant constraint, particularly when aiming to generate realistic estimations for specific and complex quantum systems. These tools often overlook the critical hardware-specific characteristics essential for accurate resource estimation.

_Algorithmic Focus and Lack of Customizability_

The primary issue with AQRE and Qualtran lies in their purely algorithmic approach. These platforms are designed to optimize quantum algorithms, such as those dependent on magic state distillation (MSD), which is currently considered one of the most efficient methods. However, this focus on optimizing algorithms comes at the expense of considering the underlying hardware architecture, which is vital for producing accurate and realistic resource estimations. For example, the AQRE and Qualtran platforms lack the ability to account for specifics such as qubit movement, local device constraints, or the physical layout of qubits, which are critical factors in the architecture of quantum systems.

In our attempts to push AQRE to its limits, we quickly realized the extent of its limitations. We explored various aspects, including the potential for movement optimization and the use of iterative decoding, but found that the tool was simply not flexible enough to accommodate these considerations. For instance, the tool could not provide concrete estimations for systems like Trapped Ions and Neutral Atoms, where the architecture’s specific needs, such as the necessity of movement to save resources, are not easily captured by a tool that is hard-coded and not dynamic.

_Hard-Coded Architectures and Inflexibility_

One of the most significant challenges we encountered with these tools is their lack of flexibility. Quantum resource estimators like AQRE and BenchQ are inherently designed to optimize predefined algorithms, and they are not equipped to adjust to the dynamic needs of various quantum architectures. For example, when examining dealing with Low-Density Parity-Check (LDPC) code architectures, the limitations became apparent. The hard-coded nature of these tools meant that any deviation from the predefined architecture resulted in inefficiencies and suboptimal results. The lack of support for critical features such as movement and flip-chip architecture further highlighted the need for a more adaptable tool.

 
### **Qubit Architecture**

_Superconducting_

Synthetic qubits, particularly those based on superconducting circuits, are one of the leading platforms in quantum computing. These qubits are realized using Josephson junctions, which allow for the creation of a two-level quantum system with low decoherence times and high controllability. The architecture for superconducting qubits is highly developed, with established techniques for qubit manipulation, readout, and error correction. However, the physical infrastructure required, such as dilution refrigerators, makes scalability and qubit connectivity challenging, especially when dealing with large numbers of qubits.

_Trapped Ions_

Trapped-ion qubits leverage ions confined in electromagnetic fields as qubits, with quantum operations performed via laser pulses or microwave fields. This platform is known for high-fidelity operations and long coherence times, making it a robust option for quantum computing. The precise control over qubit interactions and the ability to implement complex algorithms are key strengths. However, scalability remains a challenge due to the physical limitations of ion trapping and the complexity of managing large ion chains.

_Cold Atoms_

Cold atom qubits are trapped by tightly focused laser beams. These atoms can be arranged in customizable patterns, allowing for flexible and highly connected qubit geometries. Optical tweezers are promising for large-scale quantum computing due to their natural scalability and the ease of reconfiguring qubit layouts. Despite these advantages, maintaining long-term coherence and achieving consistent high-fidelity operations across many qubits are significant challenges that need to be addressed.

_Comparison and Implications for QRE_

The choice of qubit architecture—synthetic versus neutral atom—has significant implications for QRE. Superconducting qubits often require careful consideration of connectivity due to their fixed physical layout, while neutral atoms offer more flexibility in qubit arrangement but face challenges in operation fidelity. A comprehensive QRE tool must account for these differences in architecture, adjusting resource estimations based on the type of qubit used, its coherence properties, and the specific operational constraints associated with each platform.

### **Requirements**

_Functional Requirements_

- **Movement Analysis:** The tool must evaluate the impact of qubit movement within the architecture.

- **Reloading:** Consideration of the frequency and impact of atom and ion reloading within quantum operations.

- **Code Teleportation:** Ability to model and assess the implications of code teleportation on qubit resources.

- **Connectivity:** Analysis of qubit connectivity and its effect on the overall architecture.

- **Degree of Parallelization:** Estimation of the impact of parallel operations on qubit usage and efficiency.

- **Universal Gateset Configuration:** The tool must accomodate the ability to implement a universal gateset using a variety of techniques, including:
    - Magic State Distillation
    - Code Switching
    - Piece-able Fault Tolerance
    - Lattice Surgery

- **Flexible QEC Code Configuration:** Users should be able to use a variety of different quantum error correction codes, such as surface, concatenated, LDPC, etc. providing a broader range of options for optimizing quantum systems.

- **Seamless Integration:** The design should facilitate easy integration of new quantum error correction codes as they are developed, maintaining the tool’s relevance as the field advances.

- **Custom Assumptions:** Flexibility to incorporate various architectural assumptions and parameters.

### **System Architecture**

_High-Level Architecture_

- **Input Module:** Accepts user-defined parameters for the quantum architecture, including movement constraints, reloading rates, and connectivity graphs.

- **Analysis Engine:** Core computational module that performs the QRE calculations, considering all the defined parameters and assumptions.

- **Output Module:** Generates reports and visualizations, providing insights into qubit resource usage and architectural trade-offs.

- **User Interface:** A GUI that allows users to interact with the tool, input data, run simulations, and view results.

_Detailed Architecture_

- **Gateset Implementation Subsystem:** Handles the implementation and execution of quantum gates, optimizing the overall gate performance by incorporating:

	- **Gateset Selection and Optimization:** Determines the optimal gateset for specific algorithms, ensuring gate operations are executed efficiently and with high fidelity.

	- **Error Correction and Fidelity:** Implements error correction codes and monitors gate fidelity to reduce quantum error rates.

	- **Gate Scheduling:** Manages the scheduling and order of quantum gate operations to minimize idle qubit times and align with architectural constraints.

	- **Movement Submodule:** Implements algorithms to estimate the impact of qubit movement and optimizes it within the context of gate operations. It ensures that qubit transitions do not introduce significant delays or errors during gate execution.

	- **Code Teleportation Submodule:** Evaluates teleportation strategies, focusing on how they can reduce qubit movement and resource consumption. This submodule ensures that teleportation is seamlessly integrated into gate operations to maintain efficiency.

- **Atom Reloading Subsystem:** Models the atom reloading process and its effect on qubit availability, ensuring that qubits are properly aligned for gate operations.

- **Connectivity Analysis Subsystem:** Analyzes qubit connectivity patterns and their impact on the architecture, ensuring that qubits are efficiently connected for optimized gate execution and reduced latency.

- **Parallelization Analysis Subsystem:** Assesses the degree of parallelization and its effect on qubit usage, ensuring that multiple gate operations can occur simultaneously, thus increasing computational throughput.

### **Conclusion**

This QRE tool aims to fill the gap in architecture-level quantum resource estimation, providing more precise and actionable insights than existing tools. With a focus on scalability, flexibility, and usability, this tool will be a valuable asset for researchers and engineers working in the field of quantum computing.
