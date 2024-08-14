# Architecture-Level Quantum Resource Estimator

## Spencer Dearman

### **Abstract**

This document outlines the design and development of an architecture-level Quantum Resource Estimation (QRE) tool. Unlike existing QRE tools, which are mainly focused on quantum algorithm-level estimations, this tool aims to address the complexities involved in quantum architecture.


### **Background and Motivation**

Existing QRE tools are limited in their ability to provide insights at the architecture level, particularly when comparing Surface Codes with LRESC. Current tools just highlight the inherent advantage of LRESC in saving qubits, a conclusion that can be reached without extensive QRE analysis. This project aims to build a more advanced QRE tool that considers other factors such as Movement, Atom Reloading, Code Teleportation, Connectivity, Degree of Parallelization, and other architectural assumptions, providing more meaningful and actionable insights.

 
### **Current Resource Estimation Limitations**

_Limitations of Azure Quantum Resource Estimator and Qualtran_

Both the Azure Quantum Resource Estimator (AQRE) and Qualtran are prominent tools in quantum resource estimation. Despite their widespread use, these tools are primarily algorithm-centric, with limited flexibility. By default, they assume a surface quantum error correction code with lattice surgery and magic state distillation, offering no customization of these parameters. In our research, we identified alternative QEC configurations that diverge from MSD and lattice surgery; however, testing these configurations was not feasible due to the tools’ rigid, hard-coded options.

This algorithmic focus imposes a significant constraint, particularly when aiming to generate realistic estimations for specific and complex quantum systems. These tools often overlook the critical hardware-specific characteristics essential for accurate resource estimation.

_Algorithmic Focus and Lack of Customizability_

The primary issue with AQRE and Qualtran lies in their purely algorithmic approach. These platforms are designed to optimize quantum algorithms, such as those dependent on magic state distillation (MSD), which is currently considered one of the most efficient methods. However, this focus on optimizing algorithms comes at the expense of considering the underlying hardware architecture, which is vital for producing accurate and realistic resource estimations. For example, the AQRE and Qualtran platforms lack the ability to account for specifics such as qubit movement, local device constraints, or the physical layout of qubits, which are critical factors in the architecture of quantum systems.

In our attempts to push AQRE to its limits, we quickly realized the extent of its limitations. We explored various aspects, including the potential for movement optimization and the use of iterative decoding, but found that the tool was simply not flexible enough to accommodate these considerations. For instance, the tool could not provide concrete estimations for systems like Trapped Ions and Neutral Atoms, where the architecture’s specific needs, such as the necessity of movement to save resources, are not easily captured by a tool that is hard-coded and not dynamic.

_Hard-Coded Architectures and Inflexibility_

One of the most significant challenges we encountered with these tools is their lack of flexibility. Quantum resource estimators like AQRE and BenchQ are inherently designed to optimize predefined algorithms, and they are not equipped to adjust to the dynamic needs of various quantum architectures. For example, when examining the Alice and Bob architecture, or when dealing with Low-Density Parity-Check (LDPC) codes, the limitations became apparent. The hard-coded nature of these tools meant that any deviation from the predefined architecture resulted in inefficiencies and suboptimal results. The lack of support for critical features such as movement and flip-chip architecture further highlighted the need for a more adaptable tool.

 
### **Qubit Architecture**

_Superconducting_

Synthetic qubits, particularly those based on superconducting circuits, are one of the leading platforms in quantum computing. These qubits are realized using Josephson junctions, which allow for the creation of a two-level quantum system with low decoherence times and high controllability. The architecture for superconducting qubits is highly developed, with established techniques for qubit manipulation, readout, and error correction. However, the physical infrastructure required, such as dilution refrigerators, makes scalability and qubit connectivity challenging, especially when dealing with large numbers of qubits.

_Trapped Ions_

Trapped-ion qubits leverage ions confined in electromagnetic fields as qubits, with quantum operations performed via laser pulses or microwave fields. This platform is known for high-fidelity operations and long coherence times, making it a robust option for quantum computing. The precise control over qubit interactions and the ability to implement complex algorithms are key strengths. However, scalability remains a challenge due to the physical limitations of ion trapping and the complexity of managing large ion chains.

_Optical Tweezer_

Optical tweezer qubits involve neutral atoms individually trapped by tightly focused laser beams. These atoms can be arranged in customizable patterns, allowing for flexible and highly connected qubit geometries. Optical tweezers are promising for large-scale quantum computing due to their natural scalability and the ease of reconfiguring qubit layouts. Despite these advantages, maintaining long-term coherence and achieving consistent high-fidelity operations across many qubits are significant challenges that need to be addressed.

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

- **Universal Gateset Configuration:** The tool must support the ability to implement a universal gateset without relying on magic state distillation, allowing for more versatile quantum operations.

- **Flexible QEC Code Switching:** Users should be able to switch between different quantum error correction codes, such as 3D codes, subsystem codes, concatenated codes, and piece-able fault tolerance, providing a broader range of options for optimizing quantum systems.

- **Seamless Integration:** The design should facilitate easy integration of new quantum error correction codes as they are developed, maintaining the tool’s relevance as the field advances.

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
