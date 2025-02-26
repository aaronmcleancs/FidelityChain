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
    nonce: int = 0
    hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate the hash of the block"""
        block_string = (
            f"{self.index}{self.timestamp}{self.previous_hash}"
            f"{[tx.to_dict() for tx in self.transactions]}{self.nonce}"
        )
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int) -> 'Block':
        """Mine the block to find a hash with the required difficulty"""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        return self
    
    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Block':
        block = cls(
            index=data["index"],
            timestamp=data["timestamp"],
            transactions=[Transaction.from_dict(tx) for tx in data["transactions"]],
            previous_hash=data["previous_hash"],
            nonce=data["nonce"]
        )
        block.hash = data["hash"]
        return block

class Blockchain:
    """A classical blockchain with proof-of-work consensus"""
    def __init__(self, node_id: str, difficulty: int = 4):
        self.node_id = node_id
        self.difficulty = difficulty
        self.chain: List[Block] = []
        self.transaction_pool: List[Transaction] = []
        self.create_genesis_block()
    
    def create_genesis_block(self) -> None:
        """Create the genesis block"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0"
        ).mine_block(self.difficulty)
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the latest block in the chain"""
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """Add a transaction to the pool"""
        # In a real implementation, validation would happen here
        self.transaction_pool.append(transaction)
        return True
    
    def create_candidate_block(self) -> Block:
        """Create a candidate block with transactions from the pool"""
        # In a real implementation, transaction selection would be more sophisticated
        max_transactions = min(10, len(self.transaction_pool))
        selected_transactions = self.transaction_pool[:max_transactions]
        
        # Create the block
        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=selected_transactions,
            previous_hash=self.get_latest_block().hash
        )
        
        return block
    
    def mine_candidate_block(self, block: Block) -> Block:
        """Mine a candidate block"""
        return block.mine_block(self.difficulty)
    
    def add_block(self, block: Block) -> bool:
        """Add a block to the chain after validation"""
        # Validate the block
        if block.previous_hash != self.get_latest_block().hash:
            return False
        
        if block.hash[:self.difficulty] != "0" * self.difficulty:
            return False
        
        if block.hash != block.calculate_hash():
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
            
            # Check if the hash meets the difficulty requirement
            if current.hash[:self.difficulty] != "0" * self.difficulty:
                return False
            
            # Check if the previous hash field points to the previous block
            if current.previous_hash != previous.hash:
                return False
        
        return True

# ----- Classical Consensus Node -----

class ClassicalConsensusNode:
    """A node implementing proof-of-work consensus"""
    def __init__(self, node_id: str, difficulty: int = 4):
        self.node_id = node_id
        self.blockchain = Blockchain(node_id, difficulty)
        
        # Node state
        self.candidate_block: Optional[Block] = None
        self.mining_complete = False
    
    def create_candidate_block(self) -> Block:
        """Create a candidate block for the consensus round"""
        self.candidate_block = self.blockchain.create_candidate_block()
        return self.candidate_block
    
    def mine_block(self) -> Block:
        """Mine the candidate block"""
        if self.candidate_block is None:
            raise ValueError("No candidate block available")
        
        self.candidate_block = self.blockchain.mine_candidate_block(self.candidate_block)
        self.mining_complete = True
        return self.candidate_block
    
    def update_blockchain(self, block: Block) -> bool:
        """Update the blockchain with a new block"""
        return self.blockchain.add_block(block)

class ClassicalConsensusProtocol:
    """Implements a proof-of-work consensus protocol"""
    def __init__(self, num_nodes: int = 5, difficulty: int = 4):
        self.num_nodes = num_nodes
        self.difficulty = difficulty
        self.nodes: List[ClassicalConsensusNode] = []
        
        # Create nodes
        for i in range(num_nodes):
            node_id = f"node_{i}"
            node = ClassicalConsensusNode(node_id, difficulty)
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
        # Phase 1: Candidate Block Creation
        for node in self.nodes:
            node.create_candidate_block()
        
        # Phase 2: Mining Race
        winner_id = None
        winner_block = None
        
        while winner_id is None:
            # Simulate mining in steps
            for node in self.nodes:
                if not node.mining_complete:
                    # Try mining with some probability of success
                    success = random.random() < 0.2
                    if success:
                        winner_block = node.mine_block()
                        winner_id = node.node_id
                        break
            
            # If no winner yet, continue mining
            if winner_id is None:
                time.sleep(0.01)  # Small delay to avoid high CPU usage
        
        # Phase 3: Ledger Update
        for node in self.nodes:
            if node.node_id != winner_id:
                node.update_blockchain(winner_block)
        
        return winner_id, winner_block

# ----- Usage Example -----

def run_classical_blockchain_simulation(num_nodes: int = 5, num_rounds: int = 3, difficulty: int = 4):
    """Run a simulation of the classical blockchain protocol"""
    protocol = ClassicalConsensusProtocol(num_nodes, difficulty)
    
    print(f"Initialized classical blockchain with {num_nodes} nodes")
    print(f"Mining difficulty: {difficulty}")
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Add some transactions
        protocol.add_transactions(10)
        print("Added 10 random transactions to the pool")
        
        # Run consensus
        start_time = time.time()
        winner_id, winner_block = protocol.run_consensus_round()
        end_time = time.time()
        
        # Reset mining_complete flag for all nodes
        for node in protocol.nodes:
            node.mining_complete = False
        
        # Print results
        print(f"Consensus achieved in {end_time - start_time:.4f} seconds")
        print(f"Winner node: {winner_id}")
        print(f"Winning block: #{winner_block.index} with {len(winner_block.transactions)} transactions")
        print(f"Block hash: {winner_block.hash[:10]}...")
        print(f"Nonce: {winner_block.nonce}")
        
        # Verify all nodes have the same chain
        chain_lengths = [len(node.blockchain.chain) for node in protocol.nodes]
        print(f"Chain lengths: {chain_lengths}")
        
        # Verify chains are valid
        valid_chains = [node.blockchain.validate_chain() for node in protocol.nodes]
        print(f"All chains valid: {all(valid_chains)}")

if __name__ == "__main__":
    # Run a simulation with 5 nodes, 3 consensus rounds, and difficulty 4
    run_classical_blockchain_simulation(5, 3, 4)