import argparse
import time
import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal, Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import multiprocessing
from datetime import datetime

# Import both blockchain implementations
try:
    from quantumchain import run_quantum_blockchain_simulation, QuantumConsensusProtocol
    from classicalchain import run_classical_blockchain_simulation, ClassicalConsensusProtocol
    both_available = True
except ImportError:
    both_available = False
    print("Warning: One or both blockchain implementations couldn't be imported.")
    print("Make sure both quantumchain.py and classicalchain.py are in the same directory.")

@dataclass
class SimulationResults:
    """Data class to store simulation results"""
    blockchain_type: str
    num_nodes: int
    rounds: int
    difficulty: int
    transactions_per_round: int
    transaction_size: int
    execution_time: float
    time_per_round: List[float]
    consensus_times: List[float]
    block_sizes: List[int]
    chain_valid: bool
    timestamp: str

def parse_arguments():
    parser = argparse.ArgumentParser(description="Blockchain Simulation CLI")
    
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["quantum", "classical", "both"],
        default="both",
        help="Type of blockchain simulation to run (quantum, classical, or both)"
    )
    
    parser.add_argument(
        "--nodes", 
        type=int, 
        default=5,
        help="Number of nodes in the network"
    )
    
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=3,
        help="Number of consensus rounds to run"
    )
    
    parser.add_argument(
        "--difficulty", 
        type=int, 
        default=4,
        help="Mining difficulty for classical blockchain (higher is more difficult)"
    )
    
    parser.add_argument(
        "--transactions", 
        type=int, 
        default=10,
        help="Number of transactions per round"
    )
    
    parser.add_argument(
        "--tx-size",
        type=int,
        default=1,
        help="Simulated size of transactions (affects processing time)"
    )
    
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Run a performance comparison between quantum and classical"
    )
    
    parser.add_argument(
        "--batch-test",
        action="store_true",
        help="Run a batch of tests with varying parameters"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results and charts"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run simulations in parallel (faster but less detailed output)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization charts of the results"
    )
    
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat each simulation for statistical analysis"
    )
    
    parser.add_argument(
        "--scientific-report",
        action="store_true",
        help="Generate a comprehensive scientific report of results"
    )
    
    return parser.parse_args()

def interactive_mode():
    """Run an interactive CLI for blockchain simulation"""
    print("=" * 50)
    print("Blockchain Simulation Interactive CLI")
    print("=" * 50)
    
    # Check which simulations are available
    if not both_available:
        print("Error: Required modules are missing. Please check your installation.")
        return
    
    # Get simulation type
    print("\nSelect simulation type:")
    print("1. Quantum Blockchain")
    print("2. Classical Blockchain")
    print("3. Both (for comparison)")
    print("4. Batch Testing")
    
    while True:
        try:
            choice = int(input("Enter choice (1-4): "))
            if 1 <= choice <= 4:
                break
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")
    
    sim_type = ["quantum", "classical", "both", "batch"][choice-1]
    
    # Get simulation parameters
    if sim_type != "batch":
        nodes = int(input("\nNumber of nodes (default 5): ") or "5")
        rounds = int(input("Number of consensus rounds (default 3): ") or "3")
        
        if sim_type in ["classical", "both"]:
            difficulty = int(input("Mining difficulty for classical blockchain (default 4): ") or "4")
        else:
            difficulty = 4
        
        transactions = int(input("Transactions per round (default 10): ") or "10")
        tx_size = int(input("Transaction size multiplier (default 1): ") or "1")
        
        visualize = input("Generate visualization charts? (y/n, default n): ").lower() == 'y'
        
        # Run simulations
        results = run_simulations(
            sim_type, 
            nodes, 
            rounds, 
            difficulty, 
            transactions,
            tx_size,
            visualize=visualize
        )
        
        if visualize and results:
            generate_charts(results, "interactive_results")
    else:
        # Batch testing
        print("\n--- Batch Testing Configuration ---")
        node_range = input("Node range (min,max,step) default 3,10,1: ") or "3,10,1"
        min_nodes, max_nodes, step_nodes = map(int, node_range.split(','))
        
        rounds = int(input("Number of rounds (default 3): ") or "3")
        
        difficulty_range = input("Difficulty range (min,max,step) default 2,5,1: ") or "2,5,1"
        min_diff, max_diff, step_diff = map(int, difficulty_range.split(','))
        
        tx_range = input("Transactions range (min,max,step) default 5,20,5: ") or "5,20,5"
        min_tx, max_tx, step_tx = map(int, tx_range.split(','))
        
        repetitions = int(input("Repetitions for statistical validity (default 3): ") or "3")
        
        parallel = input("Run in parallel? (y/n, default y): ").lower() != 'n'
        
        visualize = input("Generate visualization charts? (y/n, default y): ").lower() != 'n'
        
        scientific_report = input("Generate scientific report? (y/n, default y): ").lower() != 'n'
        
        # Run batch tests
        all_results = run_batch_tests(
            min_nodes, max_nodes, step_nodes,
            rounds,
            min_diff, max_diff, step_diff,
            min_tx, max_tx, step_tx,
            repetitions=repetitions,
            parallel=parallel
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"batch_results_{timestamp}"
        
        save_results(all_results, output_dir)
        
        if visualize:
            generate_batch_charts(all_results, output_dir)
        
        if scientific_report:
            generate_scientific_report(all_results, output_dir)

def run_quantum_simulation_with_metrics(
    nodes: int, 
    rounds: int, 
    transactions: int,
    tx_size: int = 1
) -> SimulationResults:
    """Run quantum simulation and collect detailed metrics"""
    protocol = QuantumConsensusProtocol(nodes)
    
    print(f"Initialized quantum blockchain with {nodes} nodes")
    
    round_times = []
    consensus_times = []
    block_sizes = []
    
    start_total = time.time()
    
    for round_num in range(rounds):
        round_start = time.time()
        print(f"\n--- Round {round_num + 1} ---")
        
        # Add transactions
        protocol.add_transactions(transactions)
        
        # Simulate transaction size processing overhead
        if tx_size > 1:
            time.sleep(0.001 * transactions * tx_size)  # Simulated processing time
        
        print(f"Added {transactions} transactions to the pool")
        
        # Run consensus
        consensus_start = time.time()
        winner_id, winner_block = protocol.run_consensus_round()
        consensus_end = time.time()
        
        consensus_time = consensus_end - consensus_start
        consensus_times.append(consensus_time)
        
        # Store block size (transaction count)
        block_sizes.append(len(winner_block.transactions))
        
        # Print results
        print(f"Consensus achieved in {consensus_time:.4f} seconds")
        print(f"Winner node: {winner_id}")
        print(f"Winning block: #{winner_block.index} with {len(winner_block.transactions)} transactions")
        print(f"Block hash: {winner_block.hash[:10]}...")
        
        # Verify all nodes have the same chain
        chain_lengths = [len(node.blockchain.chain) for node in protocol.nodes]
        print(f"Chain lengths: {chain_lengths}")
        
        # Verify chains are valid
        valid_chains = [node.blockchain.validate_chain() for node in protocol.nodes]
        all_valid = all(valid_chains)
        print(f"All chains valid: {all_valid}")
        
        round_end = time.time()
        round_time = round_end - round_start
        round_times.append(round_time)
        print(f"Round {round_num + 1} completed in {round_time:.4f} seconds")
    
    end_total = time.time()
    execution_time = end_total - start_total
    
    # Check final chain validity
    chain_valid = all([node.blockchain.validate_chain() for node in protocol.nodes])
    
    # Create results object
    results = SimulationResults(
        blockchain_type="quantum",
        num_nodes=nodes,
        rounds=rounds,
        difficulty=0,  # Not applicable for quantum
        transactions_per_round=transactions,
        transaction_size=tx_size,
        execution_time=execution_time,
        time_per_round=round_times,
        consensus_times=consensus_times,
        block_sizes=block_sizes,
        chain_valid=chain_valid,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    return results

def run_classical_simulation_with_metrics(
    nodes: int, 
    rounds: int, 
    difficulty: int,
    transactions: int,
    tx_size: int = 1
) -> SimulationResults:
    """Run classical simulation and collect detailed metrics"""
    protocol = ClassicalConsensusProtocol(nodes, difficulty)
    
    print(f"Initialized classical blockchain with {nodes} nodes")
    print(f"Mining difficulty: {difficulty}")
    
    round_times = []
    consensus_times = []
    block_sizes = []
    
    start_total = time.time()
    
    for round_num in range(rounds):
        round_start = time.time()
        print(f"\n--- Round {round_num + 1} ---")
        
        # Add transactions
        protocol.add_transactions(transactions)
        
        # Simulate transaction size processing overhead
        if tx_size > 1:
            time.sleep(0.001 * transactions * tx_size)  # Simulated processing time
        
        print(f"Added {transactions} transactions to the pool")
        
        # Run consensus
        consensus_start = time.time()
        winner_id, winner_block = protocol.run_consensus_round()
        consensus_end = time.time()
        
        consensus_time = consensus_end - consensus_start
        consensus_times.append(consensus_time)
        
        # Store block size (transaction count)
        block_sizes.append(len(winner_block.transactions))
        
        # Reset mining_complete flag for all nodes
        for node in protocol.nodes:
            node.mining_complete = False
        
        # Print results
        print(f"Consensus achieved in {consensus_time:.4f} seconds")
        print(f"Winner node: {winner_id}")
        print(f"Winning block: #{winner_block.index} with {len(winner_block.transactions)} transactions")
        print(f"Block hash: {winner_block.hash[:10]}...")
        print(f"Nonce: {winner_block.nonce}")
        
        # Verify all nodes have the same chain
        chain_lengths = [len(node.blockchain.chain) for node in protocol.nodes]
        print(f"Chain lengths: {chain_lengths}")
        
        # Verify chains are valid
        valid_chains = [node.blockchain.validate_chain() for node in protocol.nodes]
        all_valid = all(valid_chains)
        print(f"All chains valid: {all_valid}")
        
        round_end = time.time()
        round_time = round_end - round_start
        round_times.append(round_time)
        print(f"Round {round_num + 1} completed in {round_time:.4f} seconds")
    
    end_total = time.time()
    execution_time = end_total - start_total
    
    # Check final chain validity
    chain_valid = all([node.blockchain.validate_chain() for node in protocol.nodes])
    
    # Create results object
    results = SimulationResults(
        blockchain_type="classical",
        num_nodes=nodes,
        rounds=rounds,
        difficulty=difficulty,
        transactions_per_round=transactions,
        transaction_size=tx_size,
        execution_time=execution_time,
        time_per_round=round_times,
        consensus_times=consensus_times,
        block_sizes=block_sizes,
        chain_valid=chain_valid,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    return results

def run_simulations(
    sim_type: Literal["quantum", "classical", "both"], 
    nodes: int, 
    rounds: int, 
    difficulty: int,
    transactions: int,
    tx_size: int = 1,
    visualize: bool = False
) -> List[SimulationResults]:
    """Run the specified blockchain simulations with detailed metrics"""
    results = []
    
    if sim_type in ["quantum", "both"]:
        print("\n" + "=" * 50)
        print("QUANTUM BLOCKCHAIN SIMULATION")
        print("=" * 50)
        quantum_results = run_quantum_simulation_with_metrics(nodes, rounds, transactions, tx_size)
        results.append(quantum_results)
    
    if sim_type in ["classical", "both"]:
        print("\n" + "=" * 50)
        print("CLASSICAL BLOCKCHAIN SIMULATION")
        print("=" * 50)
        classical_results = run_classical_simulation_with_metrics(nodes, rounds, difficulty, transactions, tx_size)
        results.append(classical_results)
    
    if sim_type == "both":
        print("\n" + "=" * 50)
        print("PERFORMANCE COMPARISON")
        print("=" * 50)
        quantum_time = quantum_results.execution_time
        classical_time = classical_results.execution_time
        
        print(f"Quantum simulation time:  {quantum_time:.4f} seconds")
        print(f"Classical simulation time: {classical_time:.4f} seconds")
        
        if quantum_time < classical_time:
            speedup = classical_time / quantum_time
            print(f"Quantum was {speedup:.2f}x faster than classical")
        else:
            slowdown = quantum_time / classical_time
            print(f"Quantum was {slowdown:.2f}x slower than classical")
        
        # Compare consensus times
        q_consensus_avg = sum(quantum_results.consensus_times) / len(quantum_results.consensus_times)
        c_consensus_avg = sum(classical_results.consensus_times) / len(classical_results.consensus_times)
        
        print(f"Average quantum consensus time:  {q_consensus_avg:.4f} seconds")
        print(f"Average classical consensus time: {c_consensus_avg:.4f} seconds")
        
        if q_consensus_avg < c_consensus_avg:
            consensus_speedup = c_consensus_avg / q_consensus_avg
            print(f"Quantum consensus was {consensus_speedup:.2f}x faster")
        else:
            consensus_slowdown = q_consensus_avg / c_consensus_avg
            print(f"Quantum consensus was {consensus_slowdown:.2f}x slower")
    
    return results

def run_single_test(
    sim_type: str,
    nodes: int,
    rounds: int,
    difficulty: int,
    transactions: int,
    tx_size: int,
    test_id: int
) -> Tuple[int, List[SimulationResults]]:
    """Run a single test case for batch testing"""
    print(f"Running test {test_id}: {sim_type} with {nodes} nodes, difficulty {difficulty}, {transactions} tx")
    try:
        results = run_simulations(sim_type, nodes, rounds, difficulty, transactions, tx_size, visualize=False)
        return test_id, results
    except Exception as e:
        print(f"Error in test {test_id}: {e}")
        return test_id, []

def run_batch_tests(
    min_nodes: int, max_nodes: int, step_nodes: int,
    rounds: int,
    min_diff: int, max_diff: int, step_diff: int,
    min_tx: int, max_tx: int, step_tx: int,
    repetitions: int = 3,
    parallel: bool = True
) -> List[SimulationResults]:
    """Run a batch of tests with varying parameters"""
    all_results = []
    test_cases = []
    test_id = 0
    
    # Generate all test cases
    for nodes in range(min_nodes, max_nodes + 1, step_nodes):
        for difficulty in range(min_diff, max_diff + 1, step_diff):
            for transactions in range(min_tx, max_tx + 1, step_tx):
                for rep in range(repetitions):
                    # Run for both blockchain types
                    test_cases.append((test_id, "quantum", nodes, rounds, difficulty, transactions, 1))
                    test_id += 1
                    test_cases.append((test_id, "classical", nodes, rounds, difficulty, transactions, 1))
                    test_id += 1
    
    total_tests = len(test_cases)
    print(f"Running {total_tests} test cases...")
    
    if parallel and total_tests > 1:
        # Run tests in parallel
        with multiprocessing.Pool() as pool:
            tasks = [(case[1], case[2], rounds, case[4], case[5], case[6], case[0]) for case in test_cases]
            results = pool.starmap(run_single_test, tasks)
            
            # Sort results by test_id and extract the result lists
            sorted_results = sorted(results, key=lambda x: x[0])
            for _, result_list in sorted_results:
                all_results.extend(result_list)
    else:
        # Run tests sequentially
        for test_id, sim_type, nodes, _, difficulty, transactions, tx_size in test_cases:
            test_results = run_single_test(sim_type, nodes, rounds, difficulty, transactions, tx_size, test_id)[1]
            all_results.extend(test_results)
            
            # Print progress
            progress = (test_id + 1) / total_tests * 100
            print(f"Progress: {progress:.1f}% ({test_id + 1}/{total_tests})")
    
    return all_results

def generate_charts(results: List[SimulationResults], output_dir: str):
    """Generate visualization charts from simulation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    if len(results) == 0:
        print("No results to visualize")
        return
    
    if len(results) == 2 and results[0].blockchain_type != results[1].blockchain_type:
        # We have both quantum and classical results for comparison
        quantum_result = next(r for r in results if r.blockchain_type == "quantum")
        classical_result = next(r for r in results if r.blockchain_type == "classical")
        
        # Plot execution time comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Execution time bar chart
        labels = ['Total Time', 'Avg Round Time', 'Avg Consensus Time']
        quantum_times = [
            quantum_result.execution_time,
            sum(quantum_result.time_per_round) / len(quantum_result.time_per_round),
            sum(quantum_result.consensus_times) / len(quantum_result.consensus_times)
        ]
        classical_times = [
            classical_result.execution_time,
            sum(classical_result.time_per_round) / len(classical_result.time_per_round),
            sum(classical_result.consensus_times) / len(classical_result.consensus_times)
        ]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, quantum_times, width, label='Quantum')
        ax.bar(x + width/2, classical_times, width, label='Classical')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Performance Comparison ({quantum_result.num_nodes} nodes, {quantum_result.transactions_per_round} tx)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_comparison.png")
        
        # Plot round-by-round comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rounds = range(1, len(quantum_result.consensus_times) + 1)
        ax.plot(rounds, quantum_result.consensus_times, 'b-o', label='Quantum')
        ax.plot(rounds, classical_result.consensus_times, 'r-o', label='Classical')
        
        ax.set_xlabel('Round Number')
        ax.set_ylabel('Consensus Time (seconds)')
        ax.set_title('Consensus Time per Round')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/round_comparison.png")
    
    # Generate individual charts for each result
    for result in results:
        prefix = f"{output_dir}/{result.blockchain_type}"
        
        # Plot consensus times by round
        fig, ax = plt.subplots(figsize=(10, 6))
        rounds = range(1, len(result.consensus_times) + 1)
        ax.plot(rounds, result.consensus_times, 'o-')
        ax.set_xlabel('Round Number')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'{result.blockchain_type.capitalize()} Consensus Time by Round')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{prefix}_consensus_times.png")
        
        # Plot block sizes by round
        fig, ax = plt.subplots(figsize=(10, 6))
        rounds = range(1, len(result.block_sizes) + 1)
        ax.plot(rounds, result.block_sizes, 'o-')
        ax.set_xlabel('Round Number')
        ax.set_ylabel('Number of Transactions')
        ax.set_title(f'{result.blockchain_type.capitalize()} Block Sizes by Round')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{prefix}_block_sizes.png")
    
    print(f"Charts generated in {output_dir}")

def generate_batch_charts(results: List[SimulationResults], output_dir: str):
    """Generate aggregate charts for batch test results"""
    os.makedirs(output_dir, exist_ok=True)
    
    if len(results) == 0:
        print("No results to visualize")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([
        {
            'type': r.blockchain_type,
            'nodes': r.num_nodes,
            'difficulty': r.difficulty,
            'transactions': r.transactions_per_round,
            'execution_time': r.execution_time,
            'avg_consensus_time': sum(r.consensus_times) / len(r.consensus_times) if r.consensus_times else 0,
            'avg_block_size': sum(r.block_sizes) / len(r.block_sizes) if r.block_sizes else 0
        }
        for r in results
    ])
    
    # Save the raw data
    df.to_csv(f"{output_dir}/all_results.csv", index=False)
    
    # Group by blockchain type, nodes, and difficulty
    grouped = df.groupby(['type', 'nodes', 'difficulty', 'transactions']).agg({
        'execution_time': ['mean', 'std'], 
        'avg_consensus_time': ['mean', 'std'],
        'avg_block_size': ['mean', 'std']
    }).reset_index()
    
    # Rename columns for clarity
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # 1. Plot scaling with number of nodes
    node_groups = grouped.groupby(['type', 'nodes'])
    node_summary = node_groups.agg({
        'execution_time_mean': 'mean',
        'avg_consensus_time_mean': 'mean'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter data by type
    quantum_data = node_summary[node_summary['type'] == 'quantum']
    classical_data = node_summary[node_summary['type'] == 'classical']
    
    if not quantum_data.empty:
        ax1.plot(quantum_data['nodes'], quantum_data['execution_time_mean'], 'b-o', label='Quantum')
    if not classical_data.empty:
        ax1.plot(classical_data['nodes'], classical_data['execution_time_mean'], 'r-o', label='Classical')
    
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Scaling with Number of Nodes (Total Time)')
    ax1.legend()
    ax1.grid(True)
    
    if not quantum_data.empty:
        ax2.plot(quantum_data['nodes'], quantum_data['avg_consensus_time_mean'], 'b-o', label='Quantum')
    if not classical_data.empty:
        ax2.plot(classical_data['nodes'], classical_data['avg_consensus_time_mean'], 'r-o', label='Classical')
    
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Consensus Time (s)')
    ax2.set_title('Scaling with Number of Nodes (Consensus Time)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/node_scaling.png")
    
    # 2. Plot scaling with difficulty (classical only)
    if not classical_data.empty:
        difficulty_groups = df[df['type'] == 'classical'].groupby(['difficulty'])
        difficulty_summary = difficulty_groups.agg({
            'execution_time': 'mean',
            'avg_consensus_time': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(difficulty_summary['difficulty'], difficulty_summary['avg_consensus_time'], 'r-o')
        ax.set_xlabel('Mining Difficulty')
        ax.set_ylabel('Consensus Time (s)')
        ax.set_title('Classical Blockchain: Scaling with Difficulty')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/difficulty_scaling.png")
    
    # 3. Plot scaling with transaction count
    tx_groups = grouped.groupby(['type', 'transactions'])
    tx_summary = tx_groups.agg({
        'execution_time_mean': 'mean',
        'avg_consensus_time_mean': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter data by type
    quantum_tx_data = tx_summary[tx_summary['type'] == 'quantum']
    classical_tx_data = tx_summary[tx_summary['type'] == 'classical']
    
    if not quantum_tx_data.empty:
        ax.plot(quantum_tx_data['transactions'], quantum_tx_data['avg_consensus_time_mean'], 'b-o', label='Quantum')
    if not classical_tx_data.empty:
        ax.plot(classical_tx_data['transactions'], classical_tx_data['avg_consensus_time_mean'], 'r-o', label='Classical')
    
    ax.set_xlabel('Number of Transactions')
    ax.set_ylabel('Consensus Time (s)')
    ax.set_title('Scaling with Transaction Count')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/transaction_scaling.png")
    
    # 4. Plot speedup ratio (quantum vs classical)
    if not quantum_data.empty and not classical_data.empty:
        # Merge data on nodes
        merged_data = pd.merge(
            quantum_data, 
            classical_data,
            on='nodes',
            suffixes=('_quantum', '_classical')
        )
        
        if not merged_data.empty:
            merged_data['speedup_ratio'] = merged_data['execution_time_mean_classical'] / merged_data['execution_time_mean_quantum']
            merged_data['consensus_speedup'] = merged_data['avg_consensus_time_mean_classical'] / merged_data['avg_consensus_time_mean_quantum']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.plot(merged_data['nodes'], merged_data['speedup_ratio'], 'g-o')
            ax1.axhline(y=1, color='r', linestyle='--')
            ax1.set_xlabel('Number of Nodes')
            ax1.set_ylabel(ax1.set_xlabel('Number of Nodes'))
            ax1.set_ylabel('Speedup Ratio (Total Time)')
            ax1.set_title('Quantum vs Classical Speedup Ratio')
            ax1.grid(True)
            
            ax2.plot(merged_data['nodes'], merged_data['consensus_speedup'], 'g-o')
            ax2.axhline(y=1, color='r', linestyle='--')
            ax2.set_xlabel('Number of Nodes')
            ax2.set_ylabel('Speedup Ratio (Consensus Time)')
            ax2.set_title('Quantum vs Classical Consensus Speedup')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/speedup_ratio.png")
    
    print(f"Batch charts generated in {output_dir}")

def save_results(results: List[SimulationResults], output_dir: str):
    """Save simulation results to JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each result as a separate JSON file
    for i, result in enumerate(results):
        # Convert dataclass to dict
        result_dict = {
            'blockchain_type': result.blockchain_type,
            'num_nodes': result.num_nodes,
            'rounds': result.rounds,
            'difficulty': result.difficulty,
            'transactions_per_round': result.transactions_per_round,
            'transaction_size': result.transaction_size,
            'execution_time': result.execution_time,
            'time_per_round': result.time_per_round,
            'consensus_times': result.consensus_times,
            'block_sizes': result.block_sizes,
            'chain_valid': result.chain_valid,
            'timestamp': result.timestamp
        }
        
        filename = f"{output_dir}/result_{i}_{result.blockchain_type}_{result.num_nodes}nodes.json"
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    # Save summary file
    summary = {
        'total_simulations': len(results),
        'quantum_count': len([r for r in results if r.blockchain_type == 'quantum']),
        'classical_count': len([r for r in results if r.blockchain_type == 'classical']),
        'avg_quantum_time': sum([r.execution_time for r in results if r.blockchain_type == 'quantum']) / 
                          max(len([r for r in results if r.blockchain_type == 'quantum']), 1),
        'avg_classical_time': sum([r.execution_time for r in results if r.blockchain_type == 'classical']) / 
                            max(len([r for r in results if r.blockchain_type == 'classical']), 1),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}")

def generate_scientific_report(results: List[SimulationResults], output_dir: str):
    """Generate a comprehensive scientific report of the simulation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame([
        {
            'type': r.blockchain_type,
            'nodes': r.num_nodes,
            'difficulty': r.difficulty,
            'transactions': r.transactions_per_round,
            'execution_time': r.execution_time,
            'avg_round_time': sum(r.time_per_round) / len(r.time_per_round) if r.time_per_round else 0,
            'avg_consensus_time': sum(r.consensus_times) / len(r.consensus_times) if r.consensus_times else 0,
            'avg_block_size': sum(r.block_sizes) / len(r.block_sizes) if r.block_sizes else 0,
            'valid_chain': r.chain_valid
        }
        for r in results
    ])
    
    # Start building the report
    report = ["# Blockchain Simulation Scientific Report\n"]
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## Summary of Simulations\n")
    report.append(f"Total simulations run: {len(results)}\n")
    report.append(f"Quantum simulations: {len(df[df['type'] == 'quantum'])}\n")
    report.append(f"Classical simulations: {len(df[df['type'] == 'classical'])}\n")
    
    # Parameter ranges
    report.append("\n## Parameter Ranges\n")
    report.append(f"Nodes: {df['nodes'].min()} to {df['nodes'].max()}\n")
    report.append(f"Difficulty (Classical): {df[df['type'] == 'classical']['difficulty'].min()} to {df[df['type'] == 'classical']['difficulty'].max()}\n")
    report.append(f"Transactions: {df['transactions'].min()} to {df['transactions'].max()}\n")
    
    # Results summary
    report.append("\n## Performance Summary\n")
    
    # Group by type
    type_summary = df.groupby('type').agg({
        'execution_time': ['mean', 'std', 'min', 'max'],
        'avg_consensus_time': ['mean', 'std', 'min', 'max']
    })
    
    report.append("### Overall Performance\n")
    report.append("| Blockchain Type | Avg Execution Time (s) | Std Dev | Min | Max | Avg Consensus Time (s) | Std Dev | Min | Max |\n")
    report.append("|----------------|------------------------|---------|-----|-----|-------------------------|---------|-----|-----|\n")
    
    for blockchain_type in ['quantum', 'classical']:
        if blockchain_type in type_summary.index:
            data = type_summary.loc[blockchain_type]
            report.append(f"| {blockchain_type.capitalize()} | {data['execution_time']['mean']:.4f} | {data['execution_time']['std']:.4f} | {data['execution_time']['min']:.4f} | {data['execution_time']['max']:.4f} | {data['avg_consensus_time']['mean']:.4f} | {data['avg_consensus_time']['std']:.4f} | {data['avg_consensus_time']['min']:.4f} | {data['avg_consensus_time']['max']:.4f} |\n")
    
    # Node scaling analysis
    report.append("\n## Scaling Analysis\n")
    
    report.append("### Scaling with Number of Nodes\n")
    node_scaling = df.groupby(['type', 'nodes']).agg({
        'execution_time': ['mean', 'std'],
        'avg_consensus_time': ['mean', 'std']
    }).reset_index()
    
    for blockchain_type in ['quantum', 'classical']:
        type_data = node_scaling[node_scaling['type'] == blockchain_type]
        if not type_data.empty:
            report.append(f"\n#### {blockchain_type.capitalize()} Blockchain\n")
            report.append("| Nodes | Avg Execution Time (s) | Std Dev | Avg Consensus Time (s) | Std Dev |\n")
            report.append("|-------|------------------------|---------|-------------------------|--------|\n")
            
            for _, row in type_data.iterrows():
                report.append(f"| {row['nodes']} | {row[('execution_time', 'mean')]:.4f} | {row[('execution_time', 'std')]:.4f} | {row[('avg_consensus_time', 'mean')]:.4f} | {row[('avg_consensus_time', 'std')]:.4f} |\n")
    
    # Difficulty scaling (classical only)
    classical_data = df[df['type'] == 'classical']
    if not classical_data.empty:
        report.append("\n### Scaling with Mining Difficulty (Classical Blockchain)\n")
        difficulty_scaling = classical_data.groupby('difficulty').agg({
            'execution_time': ['mean', 'std'],
            'avg_consensus_time': ['mean', 'std']
        }).reset_index()
        
        report.append("| Difficulty | Avg Execution Time (s) | Std Dev | Avg Consensus Time (s) | Std Dev |\n")
        report.append("|-----------|------------------------|---------|-------------------------|--------|\n")
        
        for _, row in difficulty_scaling.iterrows():
            report.append(f"| {row['difficulty']} | {row[('execution_time', 'mean')]:.4f} | {row[('execution_time', 'std')]:.4f} | {row[('avg_consensus_time', 'mean')]:.4f} | {row[('avg_consensus_time', 'std')]:.4f} |\n")
    
    # Transaction scaling
    report.append("\n### Scaling with Transaction Count\n")
    tx_scaling = df.groupby(['type', 'transactions']).agg({
        'execution_time': ['mean', 'std'],
        'avg_consensus_time': ['mean', 'std']
    }).reset_index()
    
    for blockchain_type in ['quantum', 'classical']:
        type_data = tx_scaling[tx_scaling['type'] == blockchain_type]
        if not type_data.empty:
            report.append(f"\n#### {blockchain_type.capitalize()} Blockchain\n")
            report.append("| Transactions | Avg Execution Time (s) | Std Dev | Avg Consensus Time (s) | Std Dev |\n")
            report.append("|-------------|------------------------|---------|-------------------------|--------|\n")
            
            for _, row in type_data.iterrows():
                report.append(f"| {row['transactions']} | {row[('execution_time', 'mean')]:.4f} | {row[('execution_time', 'std')]:.4f} | {row[('avg_consensus_time', 'mean')]:.4f} | {row[('avg_consensus_time', 'std')]:.4f} |\n")
    
    # Speedup analysis
    if len(df[df['type'] == 'quantum']) > 0 and len(df[df['type'] == 'classical']) > 0:
        report.append("\n## Quantum vs Classical Speedup Analysis\n")
        
        # Try to do a direct comparison by matching parameters
        quantum_df = df[df['type'] == 'quantum'].copy()
        classical_df = df[df['type'] == 'classical'].copy()
        
        # Rename columns for merging
        quantum_df = quantum_df.add_prefix('q_')
        classical_df = classical_df.add_prefix('c_')
        
        # Reset type columns to original values for matching
        quantum_df['nodes'] = quantum_df['q_nodes']
        classical_df['nodes'] = classical_df['c_nodes']
        quantum_df['transactions'] = quantum_df['q_transactions']
        classical_df['transactions'] = classical_df['c_transactions']
        
        # Merge on common parameters
        merged = pd.merge(
            quantum_df.groupby(['nodes', 'transactions']).agg({
                'q_execution_time': 'mean',
                'q_avg_consensus_time': 'mean'
            }).reset_index(),
            classical_df.groupby(['nodes', 'transactions']).agg({
                'c_execution_time': 'mean',
                'c_avg_consensus_time': 'mean'
            }).reset_index(),
            on=['nodes', 'transactions']
        )
        
        if not merged.empty:
            # Calculate speedup ratios
            merged['execution_speedup'] = merged['c_execution_time'] / merged['q_execution_time']
            merged['consensus_speedup'] = merged['c_avg_consensus_time'] / merged['q_avg_consensus_time']
            
            report.append("| Nodes | Transactions | Classical Time (s) | Quantum Time (s) | Speedup Ratio | Classical Consensus (s) | Quantum Consensus (s) | Consensus Speedup |\n")
            report.append("|-------|-------------|-------------------|-----------------|---------------|------------------------|----------------------|------------------|\n")
            
            for _, row in merged.iterrows():
                report.append(f"| {row['nodes']} | {row['transactions']} | {row['c_execution_time']:.4f} | {row['q_execution_time']:.4f} | {row['execution_speedup']:.2f}x | {row['c_avg_consensus_time']:.4f} | {row['q_avg_consensus_time']:.4f} | {row['consensus_speedup']:.2f}x |\n")
        
            # Average speedup
            avg_speedup = merged['execution_speedup'].mean()
            avg_consensus_speedup = merged['consensus_speedup'].mean()
            report.append(f"\nAverage execution speedup: {avg_speedup:.2f}x\n")
            report.append(f"Average consensus speedup: {avg_consensus_speedup:.2f}x\n")
            
            # Speedup analysis
            report.append("\n### Speedup Trends\n")
            if merged['execution_speedup'].corr(merged['nodes']) > 0.5:
                report.append("- Speedup increases with more nodes, suggesting quantum advantages scale better\n")
            elif merged['execution_speedup'].corr(merged['nodes']) < -0.5:
                report.append("- Speedup decreases with more nodes, suggesting classical approaches may scale better for larger networks\n")
            
            if merged['execution_speedup'].corr(merged['transactions']) > 0.5:
                report.append("- Speedup increases with more transactions, suggesting quantum advantages for higher throughput\n")
            elif merged['execution_speedup'].corr(merged['transactions']) < -0.5:
                report.append("- Speedup decreases with more transactions, suggesting classical advantages for higher throughput\n")
    
    # Reliability analysis
    report.append("\n## Reliability Analysis\n")
    valid_chains = df.groupby('type')['valid_chain'].mean() * 100
    report.append("| Blockchain Type | Valid Chain Percentage |\n")
    report.append("|----------------|-----------------------|\n")
    
    for blockchain_type in ['quantum', 'classical']:
        if blockchain_type in valid_chains.index:
            report.append(f"| {blockchain_type.capitalize()} | {valid_chains[blockchain_type]:.2f}% |\n")
    
    # Conclusions
    report.append("\n## Conclusions\n")
    
    if len(df[df['type'] == 'quantum']) > 0 and len(df[df['type'] == 'classical']) > 0:
        quantum_time = df[df['type'] == 'quantum']['execution_time'].mean()
        classical_time = df[df['type'] == 'classical']['execution_time'].mean()
        
        if quantum_time < classical_time:
            speedup = classical_time / quantum_time
            report.append(f"- Quantum blockchain was on average {speedup:.2f}x faster than classical blockchain\n")
        else:
            slowdown = quantum_time / classical_time
            report.append(f"- Quantum blockchain was on average {slowdown:.2f}x slower than classical blockchain\n")
        
        # Consensus time comparison
        q_consensus = df[df['type'] == 'quantum']['avg_consensus_time'].mean()
        c_consensus = df[df['type'] == 'classical']['avg_consensus_time'].mean()
        
        if q_consensus < c_consensus:
            consensus_speedup = c_consensus / q_consensus
            report.append(f"- Quantum consensus was {consensus_speedup:.2f}x faster on average\n")
        else:
            consensus_slowdown = q_consensus / c_consensus
            report.append(f"- Classical consensus was {consensus_slowdown:.2f}x faster on average\n")
    
    # Save the report
    with open(f"{output_dir}/scientific_report.md", 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Scientific report generated: {output_dir}/scientific_report.md")

def main():
    """Main entry point for the application"""
    args = parse_arguments()
    
    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        interactive_mode()
        return
    
    # Ensure both blockchain implementations are available
    if not both_available and args.type != "quantum" and args.type != "classical":
        print("Error: One or both blockchain implementations couldn't be imported.")
        print("Make sure both quantumchain.py and classicalchain.py are in the same directory.")
        return
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.batch_test:
        print("Running batch tests...")
        all_results = run_batch_tests(
            3, 10, 1,          # Nodes range (min, max, step)
            args.rounds,       # Rounds
            2, 5, 1,           # Difficulty range
            5, 20, 5,          # Transactions range
            repetitions=args.repeat,
            parallel=args.parallel
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or f"batch_results_{timestamp}"
        
        save_results(all_results, output_dir)
        
        if args.visualize:
            generate_batch_charts(all_results, output_dir)
        
        if args.scientific_report:
            generate_scientific_report(all_results, output_dir)
    elif args.compare:
        # Run comparison with specified parameters
        results = run_simulations(
            "both", 
            args.nodes, 
            args.rounds, 
            args.difficulty, 
            args.transactions,
            args.tx_size,
            visualize=args.visualize
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or f"comparison_results_{timestamp}"
        
        save_results(results, output_dir)
        
        if args.visualize:
            generate_charts(results, output_dir)
    else:
        # Run single simulation
        results = run_simulations(
            args.type, 
            args.nodes, 
            args.rounds, 
            args.difficulty, 
            args.transactions,
            args.tx_size,
            visualize=args.visualize
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or f"results_{timestamp}"
        
        save_results(results, output_dir)
        
        if args.visualize:
            generate_charts(results, output_dir)

if __name__ == "__main__":
    main()