import hashlib
import json
import time
from transaction import Transaction
from quantum_utils import extract_parameters_from_block

class Block:
    def __init__(self, index, transactions, timestamp, previous_hash, creator_id, nonce):
        self.index = index
        self.transactions = self._load_transactions(transactions)
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.creator_id = creator_id
        self.nonce = nonce
        self.hash = self.calculate_hash()
        self.quantum_state = None
        self.quantum_params = None

    def _load_transactions(self, transactions_data):
        loaded_tx = []
        for tx_data in transactions_data:
            if isinstance(tx_data, Transaction):
                loaded_tx.append(tx_data)
            elif isinstance(tx_data, dict):
                # Create transaction with all fields properly loaded
                tx = Transaction(
                    sender=tx_data['sender'], 
                    receiver=tx_data['receiver'], 
                    amount=tx_data['amount'],
                    id=tx_data.get('id'),  # Pass ID directly to constructor
                    timestamp=tx_data.get('timestamp')  # Pass timestamp directly to constructor
                )
                loaded_tx.append(tx)
            else:
                raise TypeError(f"Unsupported transaction data type: {type(tx_data)}")
        return loaded_tx

    def calculate_hash(self):
        block_dict = self.to_dict(include_hash=False)
        block_string = json.dumps(block_dict, sort_keys=True, default=str)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def extract_quantum_parameters(self):
        if self.hash is None:
            raise ValueError("Block hash must be calculated before extracting parameters.")
        self.quantum_params = extract_parameters_from_block(self)
        return self.quantum_params

    def to_dict(self, include_hash=True, include_state=False):
        tx_list = [tx.to_dict() for tx in self.transactions]
        data = {
            "index": self.index,
            "transactions": tx_list,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "creator_id": self.creator_id,
            "nonce": self.nonce,
        }
        if include_hash:
            data["hash"] = self.hash
        if include_state and self.quantum_state is not None:
            data["quantum_state_str"] = str(self.quantum_state)
        return data

    def __str__(self):
        tx_ids = [tx.id[:6] for tx in self.transactions]
        # Display full hashes for debugging
        prev_hash_display = self.previous_hash[:8] if len(self.previous_hash) > 8 else self.previous_hash
        hash_display = self.hash[:8] if self.hash and len(self.hash) > 8 else self.hash
        return (f"Block(Index: {self.index}, Time: {self.timestamp:.0f}, "
                f"Creator: {self.creator_id}, Nonce: {self.nonce}, "
                f"Tx: {len(self.transactions)} {tx_ids}, "
                f"PrevHash: {prev_hash_display}, Hash: {hash_display})")

    def __repr__(self):
        return str(self)