import time
import uuid
import hashlib

class Transaction:
    def __init__(self, sender, receiver, amount, id=None, timestamp=None):
        self.sender = sender
        self.receiver = receiver
        self.amount = float(amount)
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.id = id if id is not None else str(uuid.uuid4())

    def to_dict(self):
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "timestamp": self.timestamp
        }

    def __str__(self):
        return f"Tx({self.id[:6]} | {self.sender} -> {self.receiver}: {self.amount:.2f})"

    def __repr__(self):
        return str(self)

    def calculate_hash(self):
        tx_string = f"{self.sender}{self.receiver}{self.amount}{self.timestamp}{self.id}"
        return hashlib.sha256(tx_string.encode()).hexdigest()