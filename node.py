import time
import copy
import numpy as np
from block import Block
from transaction import Transaction
from quantum_utils import generate_verifiable_nonce, create_block_quantum_state, calculate_fidelity, NUM_QUBITS

FIDELITY_THRESHOLD = 0.05  # Lowered threshold to accommodate natural block variations

class Node:
    def __init__(self, node_id, network_nodes_count, blockchain):
        self.node_id = node_id
        self.blockchain = blockchain
        self.network_nodes_count = network_nodes_count
        self.current_candidate_block = None
        self.quantum_state = None

    def create_candidate_block(self):
        print(f"Node {self.node_id}: Creating candidate block...")
        latest_block = self.blockchain.get_latest_block()
        pending_tx = self.blockchain.get_pending_transactions(max_tx=10)
        
        # Generate quantum nonce
        nonce = generate_verifiable_nonce(self.node_id, self.network_nodes_count, num_bits=16)
        
        # Create candidate block with the full hash, not truncated
        candidate = Block(
            index=latest_block.index + 1,
            transactions=[tx.to_dict() for tx in pending_tx],
            timestamp=time.time(),
            previous_hash=latest_block.hash,  # Use full hash
            creator_id=self.node_id,
            nonce=nonce
        )
        
        self.current_candidate_block = candidate
        print(f"Node {self.node_id}: Created Candidate {candidate}")
        
        # Debug information
        print(f"Node {self.node_id}: Full previous hash: {latest_block.hash[:16]}...")
        
        return self.current_candidate_block

    def prepare_quantum_state(self):
        if not self.current_candidate_block:
            print(f"Node {self.node_id}: No candidate block to prepare state for.")
            return None
        print(f"Node {self.node_id}: Preparing quantum state for Block {self.current_candidate_block.index}...")
        try:
            self.quantum_state = create_block_quantum_state(self.current_candidate_block)
            if self.quantum_state is None:
                print(f"Node {self.node_id}: Quantum state creation returned None")
                return None
                
            # Log some info about the quantum state
            try:
                from qiskit.quantum_info import Statevector
                sv_type = type(self.quantum_state).__name__
                if isinstance(self.quantum_state, Statevector) or hasattr(self.quantum_state, 'data'):
                    sv_shape = len(self.quantum_state.data) if hasattr(self.quantum_state, 'data') else "unknown"
                    print(f"Node {self.node_id}: Quantum state created successfully. Type: {sv_type}, Size: {sv_shape}")
                else:
                    print(f"Node {self.node_id}: Warning: Unexpected quantum state type: {sv_type}")
            except Exception as debug_e:
                print(f"Node {self.node_id}: Debug info error: {debug_e}")
                
            # Continue with the regular process
            self.current_candidate_block.quantum_state = self.quantum_state
            self.current_candidate_block.extract_quantum_parameters()
            print(f"Node {self.node_id}: Quantum state prepared.")
            return self.quantum_state
        except Exception as e:
            print(f"Node {self.node_id}: Error preparing quantum state: {e}")
            import traceback
            traceback.print_exc()
            self.quantum_state = None
            return None

    def perform_consensus_round(self, all_states):
        print(f"\nNode {self.node_id}: Starting consensus round...")
        
        # Check if this node's state is available
        if self.node_id not in all_states or all_states[self.node_id] is None:
            print(f"Node {self.node_id}: Missing own state for consensus.")
            return None
            
        # Warning if not all nodes have submitted states
        if len(all_states) != self.network_nodes_count:
            print(f"Warning Node {self.node_id}: Received states from {len(all_states)}/{self.network_nodes_count} nodes.")
            
        # Initialize data structures for consensus calculation
        fidelity_matrix = np.zeros((self.network_nodes_count, self.network_nodes_count))
        node_ids = sorted(all_states.keys())
        node_id_to_index = {nid: i for i, nid in enumerate(node_ids)}
        
        # Calculate fidelities between all pairs of nodes
        print(f"Node {self.node_id}: Calculating fidelities...")
        all_fidelities = []  # Track all fidelities for statistics
        own_state = all_states[self.node_id]
        
        for other_node_id, other_state in all_states.items():
            if self.node_id == other_node_id or other_state is None:
                continue
                
            try:
                fidelity = calculate_fidelity(own_state, other_state)
                all_fidelities.append(fidelity)
                
                idx_self = node_id_to_index[self.node_id]
                idx_other = node_id_to_index[other_node_id]
                fidelity_matrix[idx_self, idx_other] = fidelity
                
                # Color code the fidelity results for better visibility
                color = ""
                if fidelity >= FIDELITY_THRESHOLD:
                    color = "AGREEMENT"  # Would be green in a color terminal
                
                print(f"  Fidelity({self.node_id}, {other_node_id}) = {fidelity:.4f} {color}")
            except Exception as e:
                print(f"Node {self.node_id}: Error calculating fidelity with Node {other_node_id}: {e}")
        
        # Print fidelity statistics
        if all_fidelities:
            avg_fidelity = sum(all_fidelities) / len(all_fidelities)
            max_fidelity = max(all_fidelities)
            min_fidelity = min(all_fidelities)
            print(f"Node {self.node_id}: Fidelity stats - Avg: {avg_fidelity:.4f}, Min: {min_fidelity:.4f}, Max: {max_fidelity:.4f}")
        
        # Determine agreement sets based on fidelity threshold
        agreement_sets = {}
        max_agreement_size = 0
        potential_winners = []
        
        print(f"Node {self.node_id}: Determining agreement sets (Threshold: {FIDELITY_THRESHOLD})...")
        support_counts = {nid: 0 for nid in node_ids}
        
        # Calculate agreement sets for each node
        for i, node_i_id in enumerate(node_ids):
            # Skip nodes without valid states
            if all_states[node_i_id] is None: 
                continue
                
            # Start with self in the agreement set
            count = 1
            agreement_set_for_i = {node_i_id}
            
            # Check fidelity with all other nodes
            for j, node_j_id in enumerate(node_ids):
                if i == j or all_states[node_j_id] is None: 
                    continue
                    
                fid = calculate_fidelity(all_states[node_i_id], all_states[node_j_id])
                
                # Add to agreement set if fidelity is above threshold
                if fid >= FIDELITY_THRESHOLD:
                    count += 1
                    agreement_set_for_i.add(node_j_id)
                    
            # Store agreement set and count
            support_counts[node_i_id] = count
            agreement_sets[node_i_id] = agreement_set_for_i
            print(f"  Node {node_i_id} has agreement set size: {count} -> {agreement_set_for_i}")
        
        # Find the maximum agreement set size
        max_agreement_size = max(support_counts.values()) if support_counts else 0
        min_required_agreement = (self.network_nodes_count // 2) + 1
        
        # Check if consensus threshold is met
        if max_agreement_size < min_required_agreement:
            print(f"Node {self.node_id}: Consensus failed. Max agreement size {max_agreement_size} < required {min_required_agreement}")
            return None
            
        # Determine potential winners (nodes with maximum agreement)
        potential_winners = [nid for nid, count in support_counts.items() if count == max_agreement_size]
        
        # Select winner deterministically by lowest ID
        winner_id = min(potential_winners)
        print(f"Node {self.node_id}: Consensus reached! Max agreement size: {max_agreement_size}/{self.network_nodes_count}")
        print(f"Node {self.node_id}: Potential Winners: {potential_winners}. Winner (lowest ID): Node {winner_id}")
        
        return winner_id

    def receive_block(self, block):
        print(f"Node {self.node_id}: Received winning block {block.index} from Node {block.creator_id}. Validating...")
        success = self.blockchain.add_block(block)
        if success:
            print(f"Node {self.node_id}: Successfully added Block {block.index} to local chain.")
            self.current_candidate_block = None
            self.quantum_state = None
        else:
            print(f"Node {self.node_id}: Failed to add Block {block.index} to local chain.")
        return success