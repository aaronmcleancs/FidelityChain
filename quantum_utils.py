import numpy as np
import hashlib
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity, Statevector

SIMULATOR = AerSimulator(
    method='statevector',
    shots=1,
    max_parallel_threads=4,
    max_memory_mb=8192
)
NUM_QUBITS = 6

def get_quantum_random_nonce(num_bits=16):
    qc = QuantumCircuit(num_bits, num_bits)
    qc.h(range(num_bits))
    qc.measure(range(num_bits), range(num_bits))
    t_qc = transpile(qc, SIMULATOR)
    circuit_name = t_qc.name
    job = SIMULATOR.run(t_qc, shots=1)
    result = job.result()
    try:
        counts = result.get_counts(circuit_name)
    except Exception:
        
        counts = result.get_counts()
    binary_string = list(counts.keys())[0]
    return int(binary_string, 2)

def get_entangled_q_RNG_bit(num_nodes):
    if num_nodes < 2:
        return get_quantum_random_nonce(num_bits=1) & 1
    qc = QuantumCircuit(num_nodes, 1)
    qc.h(0)
    for i in range(num_nodes - 1):
        qc.cx(i, i + 1)
    qc.measure(0, 0)
    t_qc = transpile(qc, SIMULATOR)
    circuit_name = t_qc.name
    job = SIMULATOR.run(t_qc, shots=1)
    result = job.result()
    try:
        counts = result.get_counts(circuit_name)
    except Exception:
        
        counts = result.get_counts()
    return int(list(counts.keys())[0])

def generate_verifiable_nonce(node_id, num_nodes, num_bits=16):
    nonce = 0
    for i in range(num_bits):
        bit = get_entangled_q_RNG_bit(num_nodes)
        nonce = (nonce << 1) | bit
    return nonce

def extract_parameters_from_block(block):
    params = {}
    
    
    tx_ids = [tx.id for tx in block.transactions]
    tx_ids.sort()  
    tx_hash_str = hashlib.sha256(','.join(tx_ids).encode()).hexdigest()
    tx_hash_bytes = bytes.fromhex(tx_hash_str)
    
    block_hash_bytes = bytes.fromhex(block.hash)
    
    params['theta'] = [(np.pi * tx_hash_bytes[i % len(tx_hash_bytes)] / 255.0) for i in range(NUM_QUBITS)]
    params['phi'] = [(2 * np.pi * tx_hash_bytes[(i + NUM_QUBITS) % len(tx_hash_bytes)] / 255.0) for i in range(NUM_QUBITS)]
    params['lambda'] = [(2 * np.pi * tx_hash_bytes[(i + 2*NUM_QUBITS) % len(tx_hash_bytes)] / 255.0) for i in range(NUM_QUBITS)]
    
    
    num_tx = len(block.transactions)
    total_volume = sum(tx.amount for tx in block.transactions) if num_tx > 0 else 0
    senders = {tx.sender for tx in block.transactions}
    diversity = len(senders) / num_tx if num_tx > 0 else 0.0
    
    params['p_tx_count'] = min(1.0, num_tx / 20.0)
    params['p_tx_volume'] = min(1.0, total_volume / 10000.0)
    params['p_diversity'] = diversity
    
    params['p_index'] = (2 / np.pi) * np.arctan(block.index / 1000.0) * 0.1  
    params['p_timestamp'] = ((block.timestamp % 3600) / 3600.0) * 0.01  
    params['p_nonce'] = ((block.nonce % (2**16)) / float(2**16 -1)) * 0.01  
    
    return params

def create_block_quantum_state(block):
    if block is None:
        raise ValueError("Block cannot be None")
    if block.hash is None:
        raise ValueError("Block hash must be calculated before state preparation")
    params = extract_parameters_from_block(block)
    qc = QuantumCircuit(NUM_QUBITS, name=f"Block_{block.index}_State")
    
    for i in range(NUM_QUBITS):
        qc.u(params['theta'][i], params['phi'][i], params['lambda'][i], i)
    qc.barrier(label="L1_Init")
    
    p_tx = (params['p_tx_count'] + params['p_tx_volume']) / 2.0
    for i in range(NUM_QUBITS):
        qc.ry(p_tx * np.pi * 3.0 * (i + 1) / NUM_QUBITS, i)
        qc.rz(params['p_diversity'] * np.pi * 2.0 * ((-1)**i), i)
    qc.barrier(label="L2_TxFeat")
    
    
    if NUM_QUBITS >= 2:
        for i in range(NUM_QUBITS - 1):
            qc.cx(i, i + 1)
    if NUM_QUBITS > 1:
        qc.cx(NUM_QUBITS - 1, 0)
    if NUM_QUBITS >= 4:
        qc.cz(0, NUM_QUBITS // 2)
    qc.barrier(label="L3_Entangle")
    
    
    for i in range(NUM_QUBITS):
        
        angle = (params['p_index'] * np.pi * (i + 1) / NUM_QUBITS +
                 params['p_timestamp'] * np.pi * (NUM_QUBITS - i) / NUM_QUBITS)
        qc.rx(angle, i)
    qc.barrier(label="L4_Struct")
    
    
    for i in range(NUM_QUBITS):
        
        qc.p(params['p_nonce'] * 2 * np.pi * ((-1)**i), i)
    qc.barrier(label="L5_Nonce")
    
    
    for i in range(NUM_QUBITS):
        qc.ry(p_tx * np.pi * (i + 1) / NUM_QUBITS, i)
    qc.barrier(label="L6_ReinforceTx")
    t_qc = transpile(qc, SIMULATOR)
    
    circuit_name = t_qc.name
    
    job = SIMULATOR.run(t_qc, shots=1, save_statevector=True)
    result = job.result()
    try:
        statevector = result.get_statevector(circuit_name)
    except Exception:
        try:
            statevector = result.data().get('statevector')
            if statevector is None:
                
                from qiskit.quantum_info import Statevector
                statevector = Statevector.from_instruction(t_qc)
        except Exception as e2:
            
            print(f"Critical error creating quantum state: {e2}")
            raise ValueError(f"Failed to get statevector")
    return statevector
def calculate_fidelity(statevector1, statevector2):
    if not isinstance(statevector1, Statevector):
        statevector1 = Statevector(statevector1)
    if not isinstance(statevector2, Statevector):
        statevector2 = Statevector(statevector2)
    return state_fidelity(statevector1, statevector2)