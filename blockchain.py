import time
from block import Block
from transaction import Transaction

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []

    def create_genesis_block(self):
        genesis_tx = Transaction("genesis", "genesis", 0)
        return Block(index=0,
                     transactions=[genesis_tx.to_dict()],
                     timestamp=1678886400,
                     previous_hash="0"*64,
                     creator_id="genesis_node",
                     nonce=0)

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, transaction):
        if not isinstance(transaction, Transaction):
            raise TypeError("Only Transaction objects can be added.")
        self.pending_transactions.append(transaction)
        print(f"Blockchain: Added pending {transaction}")
        return True

    def add_block(self, block):
        if not isinstance(block, Block):
            raise TypeError("Only Block objects can be added.")

        latest_block = self.get_latest_block()
        if block.previous_hash != latest_block.hash:
            print(f"Error: Invalid previous hash for Block {block.index}.")
            print(f"Expected: {latest_block.hash}, Got: {block.previous_hash}")
            return False

        current_hash = block.hash
        block.hash = None
        recalculated_hash = block.calculate_hash()
        block.hash = current_hash

        if current_hash != recalculated_hash:
            print(f"Error: Invalid block hash for Block {block.index}.")
            print(f"Claimed: {current_hash}, Recalculated: {recalculated_hash}")
            return False

        if block.index != latest_block.index + 1:
            print(f"Error: Invalid block index for Block {block.index}. Expected {latest_block.index + 1}")
            return False

        self.chain.append(block)
        print(f"Blockchain: Added {block}")
        block_tx_ids = {tx.id for tx in block.transactions}
        self.pending_transactions = [tx for tx in self.pending_transactions if tx.id not in block_tx_ids]
        return True

    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            current_hash = current_block.hash
            current_block.hash = None
            recalculated_hash = current_block.calculate_hash()
            current_block.hash = current_hash
            if current_hash != recalculated_hash:
                print(f"Chain Error: Hash mismatch in Block {current_block.index}")
                return False
            if current_block.previous_hash != previous_block.hash:
                print(f"Chain Error: Previous hash link broken at Block {current_block.index}")
                return False
        print("Blockchain: Chain validation successful.")
        return True

    def get_pending_transactions(self, max_tx=10):
        return self.pending_transactions[:max_tx]