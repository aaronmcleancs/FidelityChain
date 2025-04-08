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
        # Create a single shared blockchain instance for the network
        self.blockchain = Blockchain()
        # Ensure each node references the same blockchain
        self.nodes = [Node(node_id=i, network_nodes_count=num_nodes, blockchain=self.blockchain) for i in range(num_nodes)]
        
        # Get the genesis block hash for debugging
        genesis_block = self.blockchain.get_latest_block()
        genesis_hash = genesis_block.hash
        
        print(f"Network: Initialized with {num_nodes} nodes.")
        print(f"Network: Genesis Block -> {genesis_block}")
        print(f"Network: Genesis Block Hash -> {genesis_hash}")
        print(f"Network: Blockchain ID: {id(self.blockchain)} (for debugging)")
        
        # Verify all nodes share the same blockchain reference and genesis block
        for i, node in enumerate(self.nodes):
            if id(node.blockchain) != id(self.blockchain):
                print(f"Warning: Node {i} has a different blockchain reference")
            
            # Extra verification that all nodes have the same genesis block hash
            if node.blockchain.chain[0].hash != genesis_hash:
                print(f"ERROR: Node {i} has a different genesis block hash!")

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
        
        # Check genesis block hash consistency
        genesis_hashes = {}
        for node_id, node in enumerate(self.nodes):
            if node.blockchain.chain:
                genesis_hash = node.blockchain.chain[0].hash
                if genesis_hash not in genesis_hashes:
                    genesis_hashes[genesis_hash] = []
                genesis_hashes[genesis_hash].append(node_id)
        
        if len(genesis_hashes) > 1:
            print(f"WARNING: Multiple genesis block hashes detected: {len(genesis_hashes)}")
            for hash_value, node_ids in genesis_hashes.items():
                print(f"  Genesis hash {hash_value[:8]}... is used by nodes: {node_ids}")
        
        # verify blockchain consistency before starting consensus
        chain_lengths = [len(node.blockchain.chain) for node in self.nodes]
        if len(set(chain_lengths)) > 1:
            print(f"Warning: Blockchain inconsistency detected before consensus! Syncing all nodes...")
            print(f"Chain lengths: {chain_lengths}")
            # sync all nodes to the network blockchain 
            for node_id, node in enumerate(self.nodes):
                if len(node.blockchain.chain) != len(self.blockchain.chain):
                    print(f"Pre-consensus: Syncing Node {node_id} blockchain (length {len(node.blockchain.chain)}) to network blockchain (length {len(self.blockchain.chain)})")
                    
                    # check if main blockchain is empty and fix it if needed
                    if not self.blockchain.chain:
                        print(f"CRITICAL ERROR: Main network blockchain is empty before sync! This should never happen.")
                        print(f"Recreating genesis block for the network blockchain...")
                        self.blockchain.chain = [self.blockchain.create_genesis_block()]
                    
                    # create a new chain first
                    new_chain = []
                    for block in self.blockchain.chain:
                        new_chain.append(copy.deepcopy(block))
                        
                    # now assign the new chain
                    print(f"Pre-consensus: Updating Node {node_id}'s blockchain from length {len(node.blockchain.chain)} to {len(new_chain)}")
                    node.blockchain.chain = new_chain
        
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
                
                # add the winning block to the main network blockchain
                winner_node = self.nodes[winner_node_id]

                print(f"Network: Main blockchain length before adding winning block: {len(self.blockchain.chain)}")
                
                if not self.blockchain.chain:
                    print("CRITICAL ERROR: Main network blockchain is empty before adding winning block!")
                    print("Recreating genesis block for the network blockchain...")
                    self.blockchain.chain = [self.blockchain.create_genesis_block()]
                
                if self.blockchain.add_block(copy.deepcopy(winning_block)):
                    print(f"Network: Successfully added winning block to main network blockchain")
                    print(f"Network: Main blockchain length after adding block: {len(self.blockchain.chain)}")
                    
                    for node in self.nodes:
                        new_chain = []
                        for block in self.blockchain.chain:
                            new_chain.append(copy.deepcopy(block))

                        print(f"Network: Updating Node {node.node_id}'s blockchain from length {len(node.blockchain.chain)} to {len(new_chain)}")
                        node.blockchain.chain = new_chain
                        print(f"Network: Synchronized Node {node.node_id}'s blockchain with the network blockchain")
                        successful_updates += 1
                else:
                    print(f"Network: Warning - Main network blockchain rejected the winning block")
                    
                    for node in self.nodes:
                        new_chain = []
                        for block in self.blockchain.chain:
                            new_chain.append(copy.deepcopy(block))
                            
                        print(f"Network: Updating Node {node.node_id}'s blockchain from length {len(node.blockchain.chain)} to {len(new_chain)}")
                        node.blockchain.chain = new_chain
                        print(f"Network: Synchronized Node {node.node_id}'s blockchain with the network blockchain")
                        successful_updates += 1
                    
                print(f"Network: Block synchronized to {successful_updates}/{self.num_nodes} nodes.")
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
        
        print("Verifying initial blockchain consistency...")
        genesis_hashes = {}
        for node_id, node in enumerate(self.nodes):
            genesis_hash = node.blockchain.chain[0].hash
            if genesis_hash not in genesis_hashes:
                genesis_hashes[genesis_hash] = []
            genesis_hashes[genesis_hash].append(node_id)
        
        if len(genesis_hashes) > 1:
            print(f"WARNING: Multiple genesis block hashes detected at start: {len(genesis_hashes)}")
            for hash_value, node_ids in genesis_hashes.items():
                print(f"  Genesis hash {hash_value} is used by nodes: {node_ids}")
        else:
            print(f"All nodes share the same genesis block hash: {list(genesis_hashes.keys())[0]}")
        
        for i in range(num_rounds):
            print(f"\n--- Round {i+1} / {num_rounds} ---")
            
            # Verify blockchain consistency before round
            print(f"Pre-round Chain Length: {len(self.blockchain.chain)}")
            print(f"Pre-round Latest Block: {self.blockchain.get_latest_block()}")
            
            # Check integrity of the chain across all nodes
            chain_hashes = [(node_id, [block.hash for block in node.blockchain.chain]) 
                           for node_id, node in enumerate(self.nodes)]
            
            # Print hash of the latest block for each node
            print("Latest blocks for each node:")
            for node_id, hashes in chain_hashes:
                if hashes:
                    print(f"  Node {node_id}: Block {len(hashes)-1} has hash {hashes[-1]}")
                else:
                    print(f"  Node {node_id}: Empty blockchain!")
            
            # Generate transactions and run consensus
            self.generate_random_transactions(count=tx_per_round)
            success = self.run_consensus_round()
            
            # Show round results
            if success:
                print(f"--- Round {i+1} successful ---")
            else:
                print(f"--- Round {i+1} failed to reach consensus ---")
            
            # Verify the main blockchain is not empty
            if not self.blockchain.chain:
                print("CRITICAL ERROR: Main blockchain is empty after round! Recreating genesis block...")
                self.blockchain.chain = [self.blockchain.create_genesis_block()]
                
            # Verify blockchain state after round
            print(f"Current Chain Length: {len(self.blockchain.chain)}")
            print(f"Latest Block: {self.blockchain.get_latest_block()}")
            
            # Additional diagnostic - dump blockchain contents
            print(f"Main blockchain contents:")
            for block_index, block in enumerate(self.blockchain.chain):
                print(f"  Block {block_index}: {block.hash} (created by Node {block.creator_id})")
            
            # Check blockchain consistency across nodes
            chain_lengths = [len(node.blockchain.chain) for node in self.nodes]
            if len(set(chain_lengths)) > 1:
                print(f"Warning: Blockchain inconsistency detected across nodes!")
                print(f"Chain lengths: {chain_lengths}")
                
                # Fix inconsistencies by syncing all nodes to the network blockchain
                for node_id, node in enumerate(self.nodes):
                    if len(node.blockchain.chain) != len(self.blockchain.chain):
                        print(f"Syncing Node {node_id} blockchain (length {len(node.blockchain.chain)}) to network blockchain (length {len(self.blockchain.chain)})")

                        if not self.blockchain.chain:
                            print(f"CRITICAL ERROR: Main network blockchain is empty during post-round sync! This should never happen.")
                            print(f"Recreating genesis block for the network blockchain...")
                            self.blockchain.chain = [self.blockchain.create_genesis_block()]
                        
                        # Create a new chain first
                        new_chain = []
                        for block in self.blockchain.chain:
                            new_chain.append(copy.deepcopy(block))
                            
                        # Now assign the new chain
                        print(f"Post-round: Updating Node {node_id}'s blockchain from length {len(node.blockchain.chain)} to {len(new_chain)}")
                        node.blockchain.chain = new_chain
            
            time.sleep(1)
            
        print("\n===================================")
        print("=== Simulation Finished ===")
        print("===================================")
        print("Final Blockchain State:")
        for idx, block in enumerate(self.blockchain.chain):
            print(f"Block {idx}: {block}")
        
        # Validate the final chain
        valid = self.blockchain.validate_chain()
        print(f"Chain Validation Result: {'Valid' if valid else 'Invalid'}")
        
        # Final blockchain consistency check across nodes
        chain_lengths = [len(node.blockchain.chain) for node in self.nodes]
        if len(set(chain_lengths)) > 1:
            print(f"Warning: Final blockchain inconsistency detected: {chain_lengths}")
        else:
            print(f"All nodes have consistent blockchain length: {chain_lengths[0]}")