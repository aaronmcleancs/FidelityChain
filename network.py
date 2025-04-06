import time
import random
import copy
from collections import defaultdict
from blockchain import Blockchain
from node import Node
from transaction import Transaction

class Network:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.blockchain = Blockchain()
        self.nodes = [Node(node_id=i, network_nodes_count=num_nodes, blockchain=self.blockchain) for i in range(num_nodes)]
        print(f"Network: Initialized with {num_nodes} nodes.")
        print(f"Network: Genesis Block -> {self.blockchain.get_latest_block()}")

    def generate_random_transactions(self, count=5):
        print(f"\nNetwork: Generating {count} new random transactions...")
        senders = [f"User_{random.randint(0, 100)}" for _ in range(count)]
        receivers = [f"User_{random.randint(0, 100)}" for _ in range(count)]
        for i in range(count):
            sender = senders[i]
            receiver = receivers[i]
            while sender == receiver:
                receiver = f"User_{random.randint(0, 100)}"
            amount = round(random.uniform(0.1, 100.0), 2)
            tx = Transaction(sender, receiver, amount)
            self.blockchain.add_transaction(tx)

    def run_consensus_round(self):
        print(f"\n----- Starting Consensus Round for Block {self.blockchain.get_latest_block().index + 1} -----")
        start_time = time.time()
        candidate_blocks = {}
        for node in self.nodes:
            block = node.create_candidate_block()
            if block:
                candidate_blocks[node.node_id] = block
        if not candidate_blocks:
            print("Network: No nodes created candidate blocks. Skipping round.")
            return False
        quantum_states = {}
        for node_id, block in candidate_blocks.items():
            node = self.nodes[node_id]
            state = node.prepare_quantum_state()
            if state is not None:
                quantum_states[node_id] = state
            else:
                print(f"Network: Node {node_id} failed to prepare quantum state.")
        if len(quantum_states) < (self.num_nodes // 2) + 1:
            print(f"Network: Not enough nodes ({len(quantum_states)}/{self.num_nodes}) prepared states. Consensus likely to fail.")
        consensus_node = self.nodes[0]
        winner_node_id = consensus_node.perform_consensus_round(quantum_states)
        consensus_time = time.time() - start_time
        if winner_node_id is not None:
            winning_block = candidate_blocks.get(winner_node_id)
            if winning_block:
                print(f"\nNetwork: Consensus Winner: Node {winner_node_id} with Block {winning_block.index}")
                print(f"Network: Distributing winning block to all nodes...")
                successful_updates = 0
                for node in self.nodes:
                    if node.receive_block(copy.deepcopy(winning_block)):
                        successful_updates += 1
                print(f"Network: Block added by {successful_updates}/{self.num_nodes} nodes.")
                print(f"Network: Consensus round completed in {consensus_time:.4f} seconds.")
                return True
            else:
                print(f"Network: Error - Winning node {winner_node_id} has no candidate block recorded.")
                return False
        else:
            print("Network: Consensus failed for this round.")
            print(f"Network: Round attempt took {consensus_time:.4f} seconds.")
            for node in self.nodes:
                node.current_candidate_block = None
                node.quantum_state = None
            return False

    def run_simulation(self, num_rounds=5, tx_per_round=3):
        print("\n===================================")
        print("=== Starting Blockchain Simulation ===")
        print("===================================")
        for i in range(num_rounds):
            print(f"\n--- Round {i+1} / {num_rounds} ---")
            self.generate_random_transactions(count=tx_per_round)
            success = self.run_consensus_round()
            if success:
                print(f"--- Round {i+1} successful ---")
            else:
                print(f"--- Round {i+1} failed to reach consensus ---")
            print(f"Current Chain Length: {len(self.blockchain.chain)}")
            print(f"Latest Block: {self.blockchain.get_latest_block()}")
            time.sleep(1)
        print("\n===================================")
        print("=== Simulation Finished ===")
        print("===================================")
        print("Final Blockchain State:")
        for block in self.blockchain.chain:
            print(block)
        self.blockchain.validate_chain()