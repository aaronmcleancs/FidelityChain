import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer  # Aer is now in its own package
from qiskit.primitives import Sampler  # Replace execute with Sampler
from qiskit.quantum_info import state_fidelity, Statevector
import hashlib
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
import uuid

# ----- Blockchain Data Structures -----

@dataclass
class Transaction:
    """Represents a transaction in the blockchain"""
    sender: str
    receiver: str
    amount: float
    timestamp: float
    signature: str
    
    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "signature": self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            amount=data["amount"],
            timestamp=data["timestamp"],
            signature=data["signature"]
        )

@dataclass
class Block:
    """Represents a block in the blockchain"""
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    random_numbers: List[int]
    hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate the hash of the block"""
        block_string = (
            f"{self.index}{self.timestamp}{self.previous_hash}"
            f"{[tx.to_dict() for tx in self.transactions]}{self.random_numbers}"
        )
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def finalize(self) -> 'Block':
        """Calculate the hash and return the final block"""
        self.hash = self.calculate_hash()
        return self
    
    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "previous_hash": self.previous_hash,
            "random_numbers": self.random_numbers,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Block':
        block = cls(
            index=data["index"],
            timestamp=data["timestamp"],
            transactions=[Transaction.from_dict(tx) for tx in data["transactions"]],
            previous_hash=data["previous_hash"],
            random_numbers=data["random_numbers"]
        )
        block.hash = data["hash"]
        return block

class Blockchain:
    """A blockchain with quantum-assisted consensus"""
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.chain: List[Block] = []
        self.transaction_pool: List[Transaction] = []
        self.create_genesis_block()
    
    def create_genesis_block(self) -> None:
        """Create the genesis block"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0",
            random_numbers=[0]
        ).finalize()
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the latest block in the chain"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """Add a transaction to the pool"""
        # In a real implementation, validation would happen here
        self.transaction_pool.append(transaction)
        return True
    
    def create_candidate_block(self, random_numbers: List[int]) -> Block:
        """Create a candidate block with transactions from the pool"""
        # In a real implementation, transaction selection would be more sophisticated
        max_transactions = min(10, len(self.transaction_pool))
        selected_transactions = self.transaction_pool[:max_transactions]
        
        # Create the block
        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=selected_transactions,
            previous_hash=self.get_latest_block().hash,
            random_numbers=random_numbers
        ).finalize()
        
        return block
    
    def add_block(self, block: Block) -> bool:
        """Add a block to the chain after validation"""
        # In a real implementation, more validation would happen here
        if block.previous_hash != self.get_latest_block().hash:
            return False
        
        # Remove transactions that are now in the block
        tx_ids = {tx.signature for tx in block.transactions}
        self.transaction_pool = [tx for tx in self.transaction_pool if tx.signature not in tx_ids]
        
        # Add the block to the chain
        self.chain.append(block)
        return True
    
    def validate_chain(self) -> bool:
        """Validate the integrity of the blockchain"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check if the stored hash matches the calculated hash
            if current.hash != current.calculate_hash():
                return False
            
            # Check if the previous hash field points to the previous block
            if current.previous_hash != previous.hash:
                return False
        
        return True

# ----- Quantum Components -----

class QuantumRandomNumberGenerator:
    """Quantum Random Number Generator using entanglement"""
    def __init__(self, num_nodes: int, simulator: str = 'statevector_simulator'):
        self.num_nodes = num_nodes
        self.simulator = simulator
    
    def generate_random_numbers(self, num_bits: int = 8) -> List[int]:
        """Generate a list of random numbers using quantum circuits"""
        random_numbers = []
        
        for _ in range(num_bits):
            # Create a quantum circuit for N nodes
            qreg = QuantumRegister(self.num_nodes, 'q')
            creg = ClassicalRegister(self.num_nodes, 'c')
            circuit = QuantumCircuit(qreg, creg)
            
            # Apply Hadamard gates to create superposition
            for i in range(self.num_nodes):
                circuit.h(qreg[i])
            
            # Create entanglement between qubits
            for i in range(self.num_nodes-1):
                circuit.cx(qreg[i], qreg[self.num_nodes-1])
            
            # Measure all qubits
            circuit.measure(qreg, creg)
            
            # Execute the circuit using Sampler instead of execute
            backend = Aer.get_backend(self.simulator)
            sampler = Sampler()
            job = sampler.run(circuit, shots=1)
            result = job.result()
            counts = result.quasi_dists[0]
            
            # Get the measurement outcome
            measurement = list(counts.keys())[0]
            
            # Convert the first bit to an integer
            random_bit = int(bin(measurement)[2:].zfill(self.num_nodes)[0])
            random_numbers.append(random_bit)
        
        # Combine the bits into a single integer
        random_int = 0
        for bit in random_numbers:
            random_int = (random_int << 1) | bit
        
        return [random_int]

class QuantumHashFunction:
    """Maps classical block data to unique quantum states"""
    def __init__(self, simulator: str = 'statevector_simulator'):
        self.simulator = simulator
    
    def block_to_quantum_state(self, block: Block) -> Tuple[float, float]:
        """
        Convert a block to a quantum state represented by angles theta and phi
        Returns (theta, phi) where:
        - theta is in [0, pi]
        - phi is in [0, 2*pi]
        """
        # Get the hash of the block
        block_hash = block.calculate_hash()
        
        # Use the first 16 chars for theta and the last 16 chars for phi
        theta_hex = block_hash[:16]
        phi_hex = block_hash[16:32]
        
        # Convert hex to int and then to angles
        theta_int = int(theta_hex, 16)
        phi_int = int(phi_hex, 16)
        
        # Normalize to the correct ranges
        theta = (theta_int / (16**16)) * np.pi  # [0, pi]
        phi = (phi_int / (16**16)) * 2 * np.pi  # [0, 2*pi]
        
        return (theta, phi)
    
    def prepare_quantum_state(self, theta: float, phi: float) -> Statevector:
        """Prepare a quantum state using the angles theta and phi"""
        # Create a quantum circuit with one qubit
        qreg = QuantumRegister(1, 'q')
        circuit = QuantumCircuit(qreg)
        
        # Prepare the state |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
        circuit.ry(theta, 0)
        circuit.rz(phi, 0)
        
        # Simulate the circuit to get the statevector
        backend = Aer.get_backend(self.simulator)
        job = execute(circuit, backend)
        result = job.result()
        statevector = result.get_statevector()
        
        return statevector

class QuantumTeleportation:
    """Implements quantum teleportation for state sharing"""
    def __init__(self, simulator: str = 'statevector_simulator'):
        self.simulator = simulator
    
    def generate_bell_pair(self) -> Tuple[Statevector, int, int]:
        """Generate a Bell pair and return the entangled state"""
        # Create a quantum circuit with two qubits
        qreg = QuantumRegister(2, 'q')
        circuit = QuantumCircuit(qreg)
        
        # Create a Bell pair |Φ+⟩ = (|00⟩ + |11⟩)/√2
        circuit.h(0)
        circuit.cx(0, 1)
        
        # Simulate the circuit to get the statevector
        backend = Aer.get_backend(self.simulator)
        job = execute(circuit, backend)
        result = job.result()
        bell_pair = result.get_statevector()
        
        return bell_pair, 0, 1
    
    def teleport_state(self, state_to_teleport: Statevector) -> Tuple[Statevector, List[int]]:
        """
        Teleport a quantum state using Bell pairs
        Returns the teleported state and classical measurement results
        """
        # Create a quantum circuit with three qubits
        qreg = QuantumRegister(3, 'q')
        creg = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        # Initialize the first qubit to the state we want to teleport
        # In a real implementation, we would need to find the angles from the statevector
        # For simplicity, we'll use a state initialized with H and T gates
        circuit.h(0)
        circuit.t(0)
        
        # Create a Bell pair between qubits 1 and 2
        circuit.h(1)
        circuit.cx(1, 2)
        
        # Apply the teleportation protocol
        circuit.cx(0, 1)
        circuit.h(0)
        
        # Measure qubits 0 and 1
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        
        # Apply corrections based on measurement results
        circuit.z(2).c_if(creg[0], 1)
        circuit.x(2).c_if(creg[1], 1)
        
        # Simulate the circuit
        backend = Aer.get_backend(self.simulator)
        job = execute(circuit, backend, shots=1)
        result = job.result()
        counts = result.get_counts(circuit)
        
        # Get the measurement outcome
        measurement = list(counts.keys())[0]
        measurement_bits = [int(bit) for bit in measurement]
        
        # Get the final state of qubit 2 (teleported state)
        # In a real implementation, we would use a density matrix or partial trace
        # For simplicity, we'll create a new circuit with the correct state
        final_circuit = QuantumCircuit(1)
        final_circuit.h(0)
        final_circuit.t(0)
        
        # Apply corrections if needed (simulating what would happen after teleportation)
        if measurement_bits[0] == 1:
            final_circuit.z(0)
        if measurement_bits[1] == 1:
            final_circuit.x(0)
        
        # Get the statevector of the teleported state
        job = execute(final_circuit, backend)
        result = job.result()
        teleported_state = result.get_statevector()
        
        return teleported_state, measurement_bits

class FidelityChecker:
    """Computes fidelity between quantum states"""
    def __init__(self):
        pass
    
    def compute_fidelity(self, state1: Statevector, state2: Statevector) -> float:
        """Compute the fidelity between two quantum states"""
        return state_fidelity(state1, state2)
    
    def compute_fidelity_from_angles(self, theta1: float, phi1: float, theta2: float, phi2: float) -> float:
        """Compute the fidelity between two states represented by angles"""
        # Use the formula from the paper:
        # F = |cos(θ₁/2)cos(θ₂/2) + e^(i(φ₂-φ₁))sin(θ₁/2)sin(θ₂/2)|²
        
        term1 = np.cos(theta1/2) * np.cos(theta2/2)
        term2 = np.sin(theta1/2) * np.sin(theta2/2) * np.exp(1j * (phi2 - phi1))
        amplitude = term1 + term2
        fidelity = np.abs(amplitude) ** 2
        
        return fidelity

# ----- Quantum Consensus Protocol -----

class QuantumConsensusNode:
    """A node implementing the quantum-assisted consensus protocol"""
    def __init__(self, node_id: str, num_nodes: int):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.blockchain = Blockchain(node_id)
        self.quantum_rng = QuantumRandomNumberGenerator(num_nodes)
        self.quantum_hash = QuantumHashFunction()
        self.teleportation = QuantumTeleportation()
        self.fidelity_checker = FidelityChecker()
        
        # Node state
        self.candidate_block: Optional[Block] = None
        self.quantum_state: Optional[Tuple[float, float]] = None
        self.received_states: Dict[str, Tuple[float, float]] = {}
        self.fidelity_matrix: Dict[Tuple[str, str], float] = {}
    
    def generate_random_numbers(self) -> List[int]:
        """Generate random numbers for block creation"""
        return self.quantum_rng.generate_random_numbers()
    
    def create_candidate_block(self) -> Block:
        """Create a candidate block for the consensus round"""
        random_numbers = self.generate_random_numbers()
        self.candidate_block = self.blockchain.create_candidate_block(random_numbers)
        return self.candidate_block
    
    def prepare_quantum_state(self) -> Tuple[float, float]:
        """Map the candidate block to a quantum state"""
        if self.candidate_block is None:
            raise ValueError("No candidate block available")
        
        self.quantum_state = self.quantum_hash.block_to_quantum_state(self.candidate_block)
        return self.quantum_state
    
    def share_quantum_state(self, other_nodes: List['QuantumConsensusNode']) -> None:
        """Share this node's quantum state with other nodes via teleportation"""
        if self.quantum_state is None:
            raise ValueError("No quantum state prepared")
        
        # In a real implementation, this would use actual quantum teleportation
        # For simulation, we just share the angles directly
        for node in other_nodes:
            if node.node_id != self.node_id:
                node.receive_quantum_state(self.node_id, self.quantum_state)
    
    def receive_quantum_state(self, sender_id: str, state: Tuple[float, float]) -> None:
        """Receive a quantum state from another node"""
        self.received_states[sender_id] = state
    
    def compute_fidelities(self) -> Dict[Tuple[str, str], float]:
        """Compute fidelity between this node's state and received states"""
        if self.quantum_state is None:
            raise ValueError("No quantum state prepared")
        
        # Compute fidelity for each pair of nodes
        for sender_id, state in self.received_states.items():
            theta1, phi1 = self.quantum_state
            theta2, phi2 = state
            
            fidelity = self.fidelity_checker.compute_fidelity_from_angles(
                theta1, phi1, theta2, phi2
            )
            
            self.fidelity_matrix[(self.node_id, sender_id)] = fidelity
            self.fidelity_matrix[(sender_id, self.node_id)] = fidelity
        
        return self.fidelity_matrix
    
    def select_winner(self, all_fidelities: Dict[Tuple[str, str], float]) -> Tuple[str, str]:
        """Select the winning node pair based on highest fidelity"""
        # Find the pair with the highest fidelity
        max_fidelity = -1
        winning_pair = None
        
        for (node1, node2), fidelity in all_fidelities.items():
            if node1 != node2 and fidelity > max_fidelity:
                max_fidelity = fidelity
                winning_pair = (node1, node2)
        
        return winning_pair
    
    def update_blockchain(self, winning_node_id: str, winning_block: Block) -> bool:
        """Update the blockchain with the winning block"""
        return self.blockchain.add_block(winning_block)

class QuantumConsensusProtocol:
    """Implements the quantum-assisted consensus protocol"""
    def __init__(self, num_nodes: int = 5):
        self.num_nodes = num_nodes
        self.nodes: List[QuantumConsensusNode] = []
        
        # Create nodes
        for i in range(num_nodes):
            node_id = f"node_{i}"
            node = QuantumConsensusNode(node_id, num_nodes)
            self.nodes.append(node)
    
    def add_transactions(self, num_transactions: int = 20) -> None:
        """Add random transactions to all nodes"""
        for _ in range(num_transactions):
            sender = f"user_{random.randint(0, 100)}"
            receiver = f"user_{random.randint(0, 100)}"
            amount = random.uniform(1, 100)
            timestamp = time.time()
            signature = str(uuid.uuid4())
            
            transaction = Transaction(sender, receiver, amount, timestamp, signature)
            
            # Add to all nodes
            for node in self.nodes:
                node.blockchain.add_transaction(transaction)
    
    def run_consensus_round(self) -> Tuple[str, Block]:
        """Run a complete consensus round and return the winning node and block"""
        # Phase 1: Quantum Random Number Generation
        for node in self.nodes:
            node.generate_random_numbers()
        
        # Phase 2: Candidate Block Creation
        for node in self.nodes:
            node.create_candidate_block()
        
        # Phase 3: Quantum State Preparation
        for node in self.nodes:
            node.prepare_quantum_state()
        
        # Phase 4: Quantum Teleportation (State Sharing)
        for node in self.nodes:
            node.share_quantum_state(self.nodes)
        
        # Phase 5: Fidelity Computation
        all_fidelities = {}
        for node in self.nodes:
            node_fidelities = node.compute_fidelities()
            all_fidelities.update(node_fidelities)
        
        # Phase 6: Winner Selection
        # Use the first node to select the winner (in a real implementation, 
        # all nodes would reach the same conclusion independently)
        winning_pair = self.nodes[0].select_winner(all_fidelities)
        
        if winning_pair is None:
            raise ValueError("No winning pair selected")
        
        winner_id, _ = winning_pair
        winner_node = next(node for node in self.nodes if node.node_id == winner_id)
        winner_block = winner_node.candidate_block
        
        # Phase 7: Ledger Update
        for node in self.nodes:
            node.update_blockchain(winner_id, winner_block)
        
        return winner_id, winner_block

# ----- Usage Example -----

def run_quantum_blockchain_simulation(num_nodes: int = 5, num_rounds: int = 3):
    """Run a simulation of the quantum-assisted blockchain protocol"""
    protocol = QuantumConsensusProtocol(num_nodes)
    
    print(f"Initialized quantum blockchain with {num_nodes} nodes")
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Add some transactions
        protocol.add_transactions(10)
        print("Added 10 random transactions to the pool")
        
        # Run consensus
        start_time = time.time()
        winner_id, winner_block = protocol.run_consensus_round()
        end_time = time.time()
        
        # Print results
        print(f"Consensus achieved in {end_time - start_time:.4f} seconds")
        print(f"Winner node: {winner_id}")
        print(f"Winning block: #{winner_block.index} with {len(winner_block.transactions)} transactions")
        print(f"Block hash: {winner_block.hash[:10]}...")
        
        # Verify all nodes have the same chain
        chain_lengths = [len(node.blockchain.chain) for node in protocol.nodes]
        print(f"Chain lengths: {chain_lengths}")
        
        # Verify chains are valid
        valid_chains = [node.blockchain.validate_chain() for node in protocol.nodes]
        print(f"All chains valid: {all(valid_chains)}")

if __name__ == "__main__":
    # Run a simulation with 5 nodes and 3 consensus rounds
    run_quantum_blockchain_simulation(5, 3)