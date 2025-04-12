import multiprocessing as mp
import numpy as np
import json
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(processName)s] %(message)s')

# Define target distribution: Standard Normal
def target_distribution(x):
    """Unnormalised probability density of the target distribution"""
    return np.exp(-0.5 * x**2)

# Metropolis-Hastings Sampler
def metropolis_hastings(seed, num_samples=10000, burn_in=1000, thin=10, proposal_std=1.0):
    np.random.seed(seed)
    samples = []
    current = np.random.randn()
    accepted = 0

    for i in range(num_samples * thin + burn_in):
        proposal = current + np.random.normal(scale=proposal_std)
        acceptance_ratio = target_distribution(proposal) / target_distribution(current)
        
        if np.random.rand() < acceptance_ratio:
            current = proposal
            accepted += 1

        if i >= burn_in and i % thin == 0:
            samples.append(current)

    acceptance_rate = accepted / (num_samples * thin + burn_in)
    return {
        "seed": seed,
        "samples": samples,
        "mean": np.mean(samples),
        "std": np.std(samples),
        "acceptance_rate": acceptance_rate
    }

def run_chain(seed):
    logging.info(f"Starting MCMC chain with seed {seed}")
    start_time = time.time()
    
    result = metropolis_hastings(seed)

    elapsed = time.time() - start_time
    logging.info(f"Finished chain {seed} in {elapsed:.2f} seconds")
    return result

if __name__ == "__main__":
    seeds = [42, 1337, 2025, 9001]
    logging.info("Launching parallel MCMC chains...")

    with mp.Pool(processes=len(seeds)) as pool:
        results = pool.map(run_chain, seeds)

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Save each chain’s samples and summary separately
    for i, result in enumerate(results):
        chain_file = f"outputs/chain_{i+1}_results.json"
        with open(chain_file, "w") as f:
            json.dump(result, f, indent=2)
        logging.info(f"Saved Chain {i+1} results to {chain_file}")

    # Save overall summary
    summary = {
        "overall_mean": np.mean([r["mean"] for r in results]),
        "overall_std": np.mean([r["std"] for r in results]),
        "overall_acceptance_rate": np.mean([r["acceptance_rate"] for r in results]),
        "num_chains": len(results),
        "seeds": seeds
    }

    with open("outputs/mcmc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info("All results saved.")