import time
import copy
import numpy as np
from block import Block
from transaction import Transaction
from quantum_utils import generate_verifiable_nonce, create_block_quantum_state, calculate_fidelity, NUM_QUBITS

FIDELITY_THRESHOLD = 0.9

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
        nonce = generate_verifiable_nonce(self.node_id, self.network_nodes_count, num_bits=16)
        candidate = Block(
            index=latest_block.index + 1,
            transactions=[tx.to_dict() for tx in pending_tx],
            timestamp=time.time(),
            previous_hash=latest_block.hash,
            creator_id=self.node_id,
            nonce=nonce
        )
        self.current_candidate_block = candidate
        print(f"Node {self.node_id}: Created Candidate {candidate}")
        return self.current_candidate_block

    def prepare_quantum_state(self):
        if not self.current_candidate_block:
            print(f"Node {self.node_id}: No candidate block to prepare state for.")
            return None
        print(f"Node {self.node_id}: Preparing quantum state for Block {self.current_candidate_block.index}...")
        try:
            self.quantum_state = create_block_quantum_state(self.current_candidate_block)
            self.current_candidate_block.quantum_state = self.quantum_state
            self.current_candidate_block.extract_quantum_parameters()
            print(f"Node {self.node_id}: Quantum state prepared.")
            return self.quantum_state
        except Exception as e:
            print(f"Node {self.node_id}: Error preparing quantum state: {e}")
            self.quantum_state = None
            return None

    def perform_consensus_round(self, all_states):
        print(f"\nNode {self.node_id}: Starting consensus round...")
        if self.node_id not in all_states or all_states[self.node_id] is None:
            print(f"Node {self.node_id}: Missing own state for consensus.")
            return None
        if len(all_states) != self.network_nodes_count:
            print(f"Warning Node {self.node_id}: Received states from {len(all_states)}/{self.network_nodes_count} nodes.")
        fidelity_matrix = np.zeros((self.network_nodes_count, self.network_nodes_count))
        node_ids = sorted(all_states.keys())
        node_id_to_index = {nid: i for i, nid in enumerate(node_ids)}
        print(f"Node {self.node_id}: Calculating fidelities...")
        own_state = all_states[self.node_id]
        for other_node_id, other_state in all_states.items():
            if self.node_id == other_node_id or other_state is None:
                continue
            try:
                fidelity = calculate_fidelity(own_state, other_state)
                idx_self = node_id_to_index[self.node_id]
                idx_other = node_id_to_index[other_node_id]
                fidelity_matrix[idx_self, idx_other] = fidelity
                print(f"  Fidelity({self.node_id}, {other_node_id}) = {fidelity:.4f}")
            except Exception as e:
                print(f"Node {self.node_id}: Error calculating fidelity with Node {other_node_id}: {e}")
        agreement_sets = {}
        max_agreement_size = 0
        potential_winners = []
        print(f"Node {self.node_id}: Determining agreement sets (Threshold: {FIDELITY_THRESHOLD})...")
        support_counts = {nid: 0 for nid in node_ids}
        for i, node_i_id in enumerate(node_ids):
            count = 1
            if all_states[node_i_id] is None: continue
            agreement_set_for_i = {node_i_id}
            for j, node_j_id in enumerate(node_ids):
                if i == j or all_states[node_j_id] is None: continue
                fid = calculate_fidelity(all_states[node_i_id], all_states[node_j_id])
                if fid >= FIDELITY_THRESHOLD:
                    count += 1
                    agreement_set_for_i.add(node_j_id)
            support_counts[node_i_id] = count
            agreement_sets[node_i_id] = agreement_set_for_i
            print(f"  Node {node_i_id} has agreement set size: {count} -> {agreement_set_for_i}")
        max_agreement_size = max(support_counts.values()) if support_counts else 0
        min_required_agreement = (self.network_nodes_count // 2) + 1
        if max_agreement_size < min_required_agreement:
            print(f"Node {self.node_id}: Consensus failed. Max agreement size {max_agreement_size} < {min_required_agreement}")
            return None
        potential_winners = [nid for nid, count in support_counts.items() if count == max_agreement_size]
        winner_id = min(potential_winners)
        print(f"Node {self.node_id}: Consensus reached. Max agreement size: {max_agreement_size}. Potential Winners: {potential_winners}. Winner (lowest ID): Node {winner_id}")
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