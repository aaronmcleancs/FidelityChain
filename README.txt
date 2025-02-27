# Quantum-Assisted Blockchain Consensus

This project implements a blockchain system that leverages quantum computing principles to facilitate consensus among nodes, providing an alternative to traditional proof-of-work methods.

## Overview

This implementation demonstrates a quantum-assisted blockchain consensus protocol that uses quantum principles such as superposition, entanglement, and quantum state fidelity to establish agreement across a distributed network. The system includes both a quantum blockchain implementation and a classical blockchain implementation for comparison.

## Features

- **Quantum Random Number Generation**: Uses quantum circuits to generate truly random numbers
- **Quantum State Mapping**: Maps blocks to unique quantum states 
- **Quantum Teleportation**: Facilitates secure sharing of quantum states between nodes
- **Fidelity-Based Consensus**: Determines the winning block based on quantum state fidelity
- **Classical Comparison**: Includes a classical proof-of-work implementation for benchmarking

## Components

### Core Blockchain Components
- `Transaction`: Basic unit of value transfer between users
- `Block`: Container for transactions with cryptographic linkage
- `Blockchain`: Chain of blocks with validation logic

### Quantum Components
- `QuantumRandomNumberGenerator`: Produces random numbers using quantum superposition
- `QuantumHashFunction`: Maps classical data to quantum states
- `QuantumTeleportation`: Implements quantum teleportation for state sharing
- `FidelityChecker`: Computes similarity between quantum states
- `QuantumConsensusNode`: Node implementation with quantum capabilities
- `QuantumConsensusProtocol`: Coordinates the quantum consensus process

### Classical Components
- `ClassicalConsensusNode`: Node implementation with proof-of-work capabilities
- `ClassicalConsensusProtocol`: Coordinates the classical consensus process

## Requirements

- Python 3.7+
- NumPy
- Qiskit
- Qiskit Aer

## Installation

```bash
pip install numpy qiskit qiskit-aer
```

## Usage

### Running a Quantum Blockchain Simulation

```python
from quantum_blockchain import run_quantum_blockchain_simulation

# Run a simulation with 5 nodes and 3 consensus rounds
run_quantum_blockchain_simulation(5, 3)
```

### Running a Classical Blockchain Simulation

```python
from classical_blockchain import run_classical_blockchain_simulation

# Run a simulation with 5 nodes, 3 consensus rounds, and difficulty 4
run_classical_blockchain_simulation(5, 3, 4)
```

## How It Works

1. **Initialization**: Set up nodes with quantum capabilities
2. **Transaction Creation**: Generate random transactions
3. **Quantum Random Number Generation**: Each node generates random numbers using quantum circuits
4. **Candidate Block Creation**: Nodes create blocks with transactions and random numbers
5. **Quantum State Preparation**: Map each block to a unique quantum state
6. **Quantum State Sharing**: Share quantum states between nodes
7. **Fidelity Calculation**: Compute similarity metrics between quantum states
8. **Winner Selection**: Choose the winning block based on highest fidelity
9. **Blockchain Update**: All nodes update their ledger with the winning block

## Advantages Over Classical Consensus

- No computationally intensive mining required
- Potentially faster agreement time
- Quantum security properties
- Reduced energy consumption
- Natural randomness from quantum processes

## Limitations and Considerations

- Requires quantum computing resources
- Simulation currently runs on quantum simulators
- Would need quantum network infrastructure for real-world deployment
- Quantum decoherence and error correction challenges in real quantum systems

## Future Work

- Implement on real quantum hardware
- Improve quantum teleportation fidelity
- Explore additional quantum algorithms for consensus
- Analyze performance and security against quantum attacks
- Optimize for scalability with larger node networks