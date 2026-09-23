# Dyn-A-Map

# Overview

This is the official repository for the paper "Dyn-A-Map: Dynamic AI-Model Mapping for Robust and Accurate MRAM-based Analog Edge Computing". Dyn-A-Map is a novel post-training framework for variation-aware dynamic subarray mapping in SOT-MRAM-based in-memory analog computing (IMAC) edge systems.

To develop and validate our approach, we have extended the IMAC-Sim framework, a Python-based, circuit-level simulator. Our contributions are integrated into the base simulator to create a complete, fault-aware simulation and mitigation workflow. 

This repository is under active development and will continue to be updated as the framework evolves. For access to the complete set of implementation files, additional resources, or questions, please contact the authors at [mmc7@arizona.edu](mailto:mmc7@arizona.edu).

# Key Contributions: From IMAC-Sim to Dyn-A-Map

This framework's primary contributions are the modules that enable fault-aware analysis and mitigation:

## Fault Generation and Injection

Before applying Dyn-A-Map correction, the framework generates faulty weight files to simulate manufacturing variations and defects:

- **mapWB.py**: Converts clean CSV weights to resistance values and applies configurable manufacturing variations (weight_var parameter) to simulate process variations in memristive devices.
- **flip_weights_selected_subarray() function in mapLayer.py**: Selectively introduces faults by flipping resistance values (5000Ω ↔ 15000Ω) in specific subarrays with configurable flip rates (eg.70%). This simulates localized manufacturing defects or aging effects in specific crossbar regions.

## Dyn-A-Map Error Detection and Correction

1. detect_and_swap_error_subarray_with_reference.py: This is the core Dyn-A-Map algorithm. It loads faulty weights and clean reference weights, calculates a Variation Impact Score (VIS) for each subarray (FOW), and then swaps the most vulnerable (high-VIS) subarray with the most robust (low-VIS) subarray.
2. Modified testIMAC.py: Orchestrates the new workflow. It first calls mapWB.py to generate weights, then calls detect_and_swap_error_subarray_with_reference.py to correct them, and then calls mapIMAC.py to run the simulation on the corrected weights.
3. Modified mapLayer.py: This module is updated to automatically look for and load the "remapped" weight files (e.g., posweight3_remapped.txt) when building the SPICE netlist for the corrected layers.
4. error_analysis_utils.py: A helper script for analyzing and visualizing the fault distribution and the effectiveness of the swapping algorithm.
5. Enhanced Logging and Reporting: produces detailed run logs, swap histories, and summarized Excel outputs.

# Repository Structure:

```
DynAmap-IMACSim/
│
├── detect_and_swap_error_subarray_with_reference.py   # Core Dyn-A-Map module
├── error_analysis_utils.py                            # Statistical analysis helpers
├── mapIMAC.py                                         # Netlist mapping (modified from IMAC-Sim)
├── mapLayer.py                                        # Layer mapping and SPICE netlist generation (modified)
├── mapWB.py                                           # Weight/bias preprocessing; Converts .csv weights to .txt resistance files (base IMAC-Sim)
├── testIMAC.py                                        # Main experiment runner with Dyn-A-Map integration
├── data/
│   ├── W1.csv, W2.csv, W3.csv                         # Layer weight matrices
│   ├── B1.csv, B2.csv, B3.csv                         # Layer biases
│   ├── test_data.csv, test_labels.csv                 # Dataset and labels
│   ├── posweight*.txt / negweight*.txt                # Generated weight files
│   └── posweight*_reference.txt / negweight*_reference.txt  # Clean reference weights for error detection
│
├── spice/                                             # Generated SPICE netlists and subcircuits
│   ├── neuron.sp, diff*.sp, classifier.sp             # SPICE circuit models
│   ├── modelfiles/                                    # Transistor model files
│   └── layer*run*.sp                                  # Generated layer netlists
│
└── README.md                                          # This documentation file
```

Modified files: mapIMAC.py, mapLayer.py, testIMAC.py
New files: detect_and_swap_error_subarray_with_reference.py, error_analysis_utils.py

# Installation and Requirements

- Python 3.x
- NumPy
- pandas
- openpyxl
- HSPICE (must be callable via the hspice command in your terminal)
- (Optional) matplotlib for visualization of error analysis

# IMAC-Sim

IMAC-Sim is a Python-based simulation framework, which creates the SPICE netlist of the In-Memory Analog Computing (IMAC) circuit based on various device- and circuit-level hyperparameters selected by the user, and automatically evaluates the accuracy, power consumption and latency of the developed circuit using a user-specified dataset. The list of currently supported inputs are as follows.

- data_dir: The directory where data files are located
- spice_dir: The directory where spice files are located
- dataset_file: Name of the dataset file
- label_file: Name of the label file
- weight_var: percentage variation in the resistance of the synapses
- testnum: Number of input test cases to run
- testnum_per_batch: Number of test cases in a single batch
- firstimage: Starting point of the test inputs in the dataset file
- vdd: The positive supply voltage
- vss: The negative supply voltage
- tsampling: The sampling time in nanosecond
- nodes: Network Topology, an array which defines the DNN model size
- xbar: The m x n crossbar size represented as [m,n]
- gain: Array for the differential amplifier gains of all hidden layers
- tech_node: The technology node e.g. 9nm, 45nm etc.
- metal: Width of the metal line for parasitic calculation
- T: Metal thickness
- H: Inter metal layer spacing
- L: length of the bitcell
- W: width of the bitcell
- D: Distance between I+ and I- lines
- eps: Permittivity of oxide
- rho: Resistivity of metal
- rlow: Low resistance level of the memristive device
- rhigh: High resistance level of the memristive device
- hpar: Array for the horizontal partitioning of all hidden layers, calculated automatically for a given crossbar size
- vpar: Array for the vertical partitioning of all hidden layers, calculated automatically for a given crossbar size

Follow this paper for more information. Md Hasibul Amin, Mohammed E. Elbtity, and Ramtin Zand. 2023. IMAC-Sim: A Circuit-level Simulator For In-Memory Analog Computing Architectures. In Proceedings of the Great Lakes Symposium on VLSI 2023 (GLSVLSI '23). Association for Computing Machinery, New York, NY, USA, 659–664. https://doi.org/10.1145/3583781.3590264

## Running the code

- Prepare the /data Directory: First go to the directory _data_ and put the input test data, label data, pre-trained weights and biases into the directory. Sample files are provided for a 400 X 120 X 84 X 10 DNN model. To perform fault correction, you must also add the "clean" reference weight files for the layers you intend to correct (e.g., posweight3_reference.txt, negweight3_reference.txt). The detect_and_swap_error_subarray_with_reference.py script uses these files to calculate the VIS and identify faulty subarrays.
- Go to the directory _spice_ and put the neuron subcircuit, differential amplifier subcircuit and transistor models into the directory. Sample model files are provided.
- Open _testIMAC.py_ and modify the list of inputs as required. In testIMAC.py, modify the weight_var parameter to set the manufacturing variation percentage (e.g., weight_var = 5.0 for 5% variation). Additionally, you can modify the flip_rate parameter in the flip_weights_selected_subarray() function call within mapLayer.py to control the percentage of weights flipped in selected subarrays (e.g., flip_rate=0.70 for 70% flip rate).
- Make sure you have HSpice installed in your machine and accessible through _hspice_ command in the terminal. The code initiates a HSpice run through an os.system call of the _hspice_ command.
- Run the IMAC using the command _python testIMAC.py_. This will build the necessary IMAC subcircuits in the _spice_ directory, run the netlist on HSpice and print the accuracy and power consumption results of your input test cases batch-by-batch in the terminal.

## More documentation

Additional References

- Chowdhury M.A., Hossain M., Mastrangelo C., DeMara R.F., Salehi S., "S-Tune: SOT-MTJ Manufacturing Parameters Tuning for Securing the Next Generation of Computing," Frontiers in Electronics, 2024.
- M. Hossain, M. A. Chowdhury, R. F. DeMara and S. Salehi, "Sensitivity Analysis of SOT-MTJs to Manufacturing Process Variation: A Hardware Security Perspective," ISQED 2024, pp. 1–5.
- M. H. Amin, M. E. Elbtity and R. Zand, "Xbar-Partitioning: A Practical Way for Parasitics and Noise Tolerance in Analog IMC Circuits," in IEEE Journal on Emerging and Selected Topics in Circuits and Systems, vol. 12, no. 4, pp. 867-877, Dec. 2022, doi: 10.1109/JETCAS.2022.3222966.
- M. Elbtity, A. Singh, B. Reidy, X. Guo and R. Zand, "An In-Memory Analog Computing Co-Processor for Energy-Efficient CNN Inference on Mobile Devices," 2021 IEEE Computer Society Annual Symposium on VLSI (ISVLSI), Tampa, FL, USA, 2021, pp. 188-193, doi: 10.1109/ISVLSI51109.2021.00043.
- Md Hasibul Amin, Mohammed Elbtity, Mohammadreza Mohammadi, and Ramtin Zand. 2022. MRAM-based Analog Sigmoid Function for In-memory Computing. In Proceedings of the Great Lakes Symposium on VLSI 2022 (GLSVLSI '22). Association for Computing Machinery, New York, NY, USA, 319–323. https://doi.org/10.1145/3526241.3530376
- M. H. Amin, M. Elbtity and R. Zand, "Interconnect Parasitics and Partitioning in Fully-Analog In-Memory Computing Architectures," 2022 IEEE International Symposium on Circuits and Systems (ISCAS), Austin, TX, USA, 2022, pp. 389-393, doi: 10.1109/ISCAS48785.2022.9937884.

# Acknowledgment

This work extends the IMAC-Sim simulator originally developed by Md Hasibul Amin et al. (GLSVLSI 2023).
We gratefully acknowledge their open-source contribution, which served as the base framework for the Dyn-A-Map development.
