from network import Network

NUM_NODES = 5
NUM_ROUNDS = 10
TX_PER_ROUND = 5

if __name__ == "__main__":
    network = Network(num_nodes=NUM_NODES)
    network.run_simulation(num_rounds=NUM_ROUNDS, tx_per_round=TX_PER_ROUND)
    print("\nSimulation Complete.")