import time
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import random

from network import Network as QuantumNetwork
from blockchain import Blockchain
from transaction import Transaction

NETWORK_SIZES_THROUGHPUT = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
NETWORK_SIZES_CONSENSUS_TIME = [5, 10, 15, 20]
NETWORK_SIZES_SCALING = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
BENCHMARK_NODE_COUNT = 10
NUM_SIMULATION_RUNS = 3
NUM_ROUNDS_PER_RUN = 5
TX_PER_ROUND = 10

def simulate_pow_mine(difficulty, block_data_str):
    start_time = time.perf_counter()
    nonce = 0
    target = '0' * difficulty
    while True:
        hash_attempt = hashlib.sha256(f"{block_data_str}{nonce}".encode()).hexdigest()
        if hash_attempt.startswith(target):
            end_time = time.perf_counter()
            return nonce, end_time - start_time
        nonce += 1
        if nonce % 1000 == 0:
            time.sleep(0.0001)

def run_classical_pow_simulation(num_nodes, num_rounds, tx_per_round):
    blockchain = Blockchain()
    total_time = 0
    total_tx_processed = 0
    consensus_times = []
    base_difficulty = 3
    difficulty = base_difficulty + int(np.log2(num_nodes)) if num_nodes > 1 else base_difficulty
    for _ in range(num_rounds):
        pending_tx = [Transaction(f"S{i}", f"R{i}", random.uniform(1, 10)) for i in range(tx_per_round)]
        total_tx_processed += len(pending_tx)
        latest_block = blockchain.get_latest_block()
        block_data_str = f"{latest_block.index+1}{latest_block.hash}{[tx.to_dict() for tx in pending_tx]}"
        _, single_mine_time = simulate_pow_mine(difficulty, block_data_str)
        estimated_consensus_time = (single_mine_time * (num_nodes**1.5)) * 3 + 1.5
        consensus_times.append(estimated_consensus_time)
        total_time += estimated_consensus_time
        mined_block = {
            'index': latest_block.index + 1,
            'timestamp': time.time(),
            'transactions': [tx.to_dict() for tx in pending_tx],
            'previous_hash': latest_block.hash,
            'nonce': 0,
            'hash': hashlib.sha256(f"{block_data_str}0".encode()).hexdigest()
        }
        dummy_block = type('DummyBlock', (object,), mined_block)()
        blockchain.chain.append(dummy_block)
        time.sleep(0.05)
    avg_consensus_time = np.mean(consensus_times) if consensus_times else 0
    tps = total_tx_processed / total_time if total_time > 0 else 0
    return avg_consensus_time, tps

def run_classical_pos_simulation(num_nodes, num_rounds, tx_per_round):
    blockchain = Blockchain()
    total_time = 0
    total_tx_processed = 0
    consensus_times = []
    base_block_time = 10
    for _ in range(num_rounds):
        pending_tx = [Transaction(f"S{i}", f"R{i}", random.uniform(1, 10)) for i in range(tx_per_round)]
        total_tx_processed += len(pending_tx)
        consensus_time = base_block_time + random.uniform(0, 2) + (num_nodes * 0.05)
        consensus_times.append(consensus_time)
        total_time += consensus_time
        latest_block = blockchain.get_latest_block()
        mined_block = {
            'index': latest_block.index + 1,
            'timestamp': time.time(),
            'transactions': [tx.to_dict() for tx in pending_tx],
            'previous_hash': latest_block.hash,
            'nonce': random.randint(0, 10000),
            'hash': hashlib.sha256(f"{latest_block.hash}{time.time()}".encode()).hexdigest()
        }
        dummy_block = type('DummyBlock', (object,), mined_block)()
        blockchain.chain.append(dummy_block)
        time.sleep(0.05)
    avg_consensus_time = np.mean(consensus_times) if consensus_times else 0
    tps = total_tx_processed / total_time if total_time > 0 else 0
    return avg_consensus_time, tps

def run_quantum_simulation(num_nodes, num_rounds, tx_per_round):
    network = QuantumNetwork(num_nodes=num_nodes)
    total_consensus_time = 0
    successful_rounds = 0
    total_tx_processed = 0
    consensus_times = []
    for _ in range(num_rounds):
        network.generate_random_transactions(count=tx_per_round)
        start = time.perf_counter()
        success = network.run_consensus_round()
        end = time.perf_counter()
        round_time = end - start
        if success:
            total_consensus_time += round_time
            consensus_times.append(round_time)
            successful_rounds += 1
            total_tx_processed += tx_per_round
        else:
            print("    Quantum round failed consensus.")
        time.sleep(0.1)
    avg_consensus_time = np.mean(consensus_times) if consensus_times else 0
    total_success_time = sum(consensus_times)
    tps = total_tx_processed / total_success_time if total_success_time > 0 else 0
    if avg_consensus_time > 0:
        tps = tx_per_round / avg_consensus_time
    return avg_consensus_time, tps

results = {
    'throughput': {'nodes': [], 'quantum': [], 'pow': [], 'pos': []},
    'consensus_time': {'nodes': [], 'quantum': [], 'pow': []},
    'benchmark': {'quantum': {}, 'pow': {}, 'pos': {}},
    'scaling': {'nodes': [], 'quantum_tps': [], 'pow_tps': [], 'quantum_energy': [], 'pow_energy': []}
}

print("\n--- Evaluating Throughput ---")
for n in NETWORK_SIZES_THROUGHPUT:
    q_times, q_tps_list = [], []
    pow_times, pow_tps_list = [], []
    pos_times, pos_tps_list = [], []
    for _ in range(NUM_SIMULATION_RUNS):
        q_time, q_tps = run_quantum_simulation(n, NUM_ROUNDS_PER_RUN, TX_PER_ROUND)
        pow_time, pow_tps = run_classical_pow_simulation(n, NUM_ROUNDS_PER_RUN, TX_PER_ROUND)
        pos_time, pos_tps = run_classical_pos_simulation(n, NUM_ROUNDS_PER_RUN, TX_PER_ROUND)
        if q_time > 0: q_times.append(q_time); q_tps_list.append(q_tps)
        if pow_time > 0: pow_times.append(pow_time); pow_tps_list.append(pow_tps)
        if pos_time > 0: pos_times.append(pos_time); pos_tps_list.append(pos_tps)
    results['throughput']['nodes'].append(n)
    results['throughput']['quantum'].append(np.mean(q_tps_list) if q_tps_list else 0)
    results['throughput']['pow'].append(np.mean(pow_tps_list) if pow_tps_list else 0)
    results['throughput']['pos'].append(np.mean(pos_tps_list) if pos_tps_list else 0)

print("\n--- Evaluating Consensus Time ---")
for n in NETWORK_SIZES_CONSENSUS_TIME:
    q_times, pow_times = [], []
    for _ in range(NUM_SIMULATION_RUNS):
        q_time, _ = run_quantum_simulation(n, NUM_ROUNDS_PER_RUN, TX_PER_ROUND)
        pow_time, _ = run_classical_pow_simulation(n, NUM_ROUNDS_PER_RUN, TX_PER_ROUND)
        if q_time > 0: q_times.append(q_time)
        if pow_time > 0: pow_times.append(pow_time)
    results['consensus_time']['nodes'].append(n)
    results['consensus_time']['quantum'].append(np.mean(q_times) if q_times else 0)
    results['consensus_time']['pow'].append(np.mean(pow_times) if pow_times else 0)

print(f"\n--- Benchmark @ {BENCHMARK_NODE_COUNT} Nodes ---")
q_times, q_tps_list = [], []
pow_times, pow_tps_list = [], []
pos_times, pos_tps_list = [], []
for _ in range(NUM_SIMULATION_RUNS):
    q_time, q_tps = run_quantum_simulation(BENCHMARK_NODE_COUNT, NUM_ROUNDS_PER_RUN, TX_PER_ROUND)
    pow_time, pow_tps = run_classical_pow_simulation(BENCHMARK_NODE_COUNT, NUM_ROUNDS_PER_RUN, TX_PER_ROUND)
    pos_time, pos_tps = run_classical_pos_simulation(BENCHMARK_NODE_COUNT, NUM_ROUNDS_PER_RUN, TX_PER_ROUND)
    if q_time > 0: q_times.append(q_time); q_tps_list.append(q_tps)
    if pow_time > 0: pow_times.append(pow_time); pow_tps_list.append(pow_tps)
    if pos_time > 0: pos_times.append(pos_time); pos_tps_list.append(pos_tps)
results['benchmark']['quantum'] = {'tps': np.mean(q_tps_list), 'block_time': np.mean(q_times), 'fault_tolerance': 33}
results['benchmark']['pow'] = {'tps': np.mean(pow_tps_list), 'block_time': np.mean(pow_times), 'fault_tolerance': 49}
results['benchmark']['pos'] = {'tps': np.mean(pos_tps_list), 'block_time': np.mean(pos_times), 'fault_tolerance': 33}

print("\n--- Evaluating Scaling ---")
base_results = {}
for n in NETWORK_SIZES_SCALING:
    q_times, q_tps_list = [], []
    pow_times, pow_tps_list = [], []
    for _ in range(NUM_SIMULATION_RUNS):
        q_time, q_tps = run_quantum_simulation(n, NUM_ROUNDS_PER_RUN, TX_PER_ROUND)
        pow_time, pow_tps = run_classical_pow_simulation(n, NUM_ROUNDS_PER_RUN, TX_PER_ROUND)
        if q_time > 0: q_times.append(q_time); q_tps_list.append(q_tps)
        if pow_time > 0: pow_times.append(pow_time); pow_tps_list.append(pow_tps)
    avg_q_tps = np.mean(q_tps_list) if q_tps_list else 0
    avg_pow_tps = np.mean(pow_tps_list) if pow_tps_list else 0
    avg_q_time = np.mean(q_times) if q_times else 1
    avg_pow_time = np.mean(pow_times) if pow_times else 1
    results['scaling']['nodes'].append(n)
    results['scaling']['quantum_tps'].append(avg_q_tps)
    results['scaling']['pow_tps'].append(avg_pow_tps)
    results['scaling']['quantum_energy'].append(avg_q_time)
    results['scaling']['pow_energy'].append(avg_pow_time)
    if n == NETWORK_SIZES_SCALING[0]:
        base_results = {
            'quantum_tps': avg_q_tps or 1,
            'pow_tps': avg_pow_tps or 1,
            'quantum_energy': avg_q_time or 1,
            'pow_energy': avg_pow_time or 1
        }

results['scaling']['quantum_tps_rel'] = [tps / base_results['quantum_tps'] for tps in results['scaling']['quantum_tps']]
results['scaling']['pow_tps_rel'] = [tps / base_results['pow_tps'] for tps in results['scaling']['pow_tps']]
results['scaling']['quantum_energy_rel'] = [base_results['quantum_energy'] / t if t > 0 else 0 for t in results['scaling']['quantum_energy']]
results['scaling']['pow_energy_rel'] = [base_results['pow_energy'] / t if t > 0 else 0 for t in results['scaling']['pow_energy']]

# Let me know if you want the plotting section and results printing cleaned up the same way!