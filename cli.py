import argparse
import time
import sys
from typing import Literal

# Import both blockchain implementations
try:
    from quantumchain import run_quantum_blockchain_simulation
    from classicalchain import run_classical_blockchain_simulation
    both_available = True
except ImportError:
    both_available = False
    print("Warning: One or both blockchain implementations couldn't be imported.")
    print("Make sure both quantum_blockchain.py and classical_blockchain.py are in the same directory.")

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
        "--compare", 
        action="store_true",
        help="Run a performance comparison between quantum and classical"
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
    
    while True:
        try:
            choice = int(input("Enter choice (1-3): "))
            if 1 <= choice <= 3:
                break
            else:
                print("Please enter a number between 1 and 3.")
        except ValueError:
            print("Please enter a valid number.")
    
    sim_type = ["quantum", "classical", "both"][choice-1]
    
    # Get simulation parameters
    nodes = int(input("\nNumber of nodes (default 5): ") or "5")
    rounds = int(input("Number of consensus rounds (default 3): ") or "3")
    
    if sim_type in ["classical", "both"]:
        difficulty = int(input("Mining difficulty for classical blockchain (default 4): ") or "4")
    else:
        difficulty = 4
    
    transactions = int(input("Transactions per round (default 10): ") or "10")
    
    # Run simulations
    run_simulations(sim_type, nodes, rounds, difficulty, transactions)

def run_simulations(
    sim_type: Literal["quantum", "classical", "both"], 
    nodes: int, 
    rounds: int, 
    difficulty: int,
    transactions: int
):
    """Run the specified blockchain simulations"""
    if sim_type in ["quantum", "both"]:
        print("\n" + "=" * 50)
        print("QUANTUM BLOCKCHAIN SIMULATION")
        print("=" * 50)
        start_time = time.time()
        run_quantum_blockchain_simulation(nodes, rounds)
        quantum_time = time.time() - start_time
        print(f"\nTotal quantum simulation time: {quantum_time:.4f} seconds")
    
    if sim_type in ["classical", "both"]:
        print("\n" + "=" * 50)
        print("CLASSICAL BLOCKCHAIN SIMULATION")
        print("=" * 50)
        start_time = time.time()
        run_classical_blockchain_simulation(nodes, rounds, difficulty)
        classical_time = time.time() - start_time
        print(f"\nTotal classical simulation time: {classical_time:.4f} seconds")
    
    if sim_type == "both":
        print("\n" + "=" * 50)
        print("PERFORMANCE COMPARISON")
        print("=" * 50)
        print(f"Quantum simulation time:  {quantum_time:.4f} seconds")
        print(f"Classical simulation time: {classical_time:.4f} seconds")
        
        if quantum_time < classical_time:
            speedup = classical_time / quantum_time
            print(f"Quantum was {speedup:.2f}x faster than classical")
        else:
            slowdown = quantum_time / classical_time
            print(f"Quantum was {slowdown:.2f}x slower than classical")

def main():
    # Check if any arguments were provided
    if len(sys.argv) > 1:
        # Use argument parser
        args = parse_arguments()
        run_simulations(args.type, args.nodes, args.rounds, args.difficulty, args.transactions)
    else:
        # Use interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
