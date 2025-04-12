import multiprocessing as mp
import time
import json
import os

def run_chain(seed):
    """Simulate a fake MCMC chain"""
    print(f"[Chain {seed}] Starting...")
    time.sleep(2)  # Simulate computation
    result = {"seed": seed, "final_value": seed ** 2}
    print(f"[Chain {seed}] Done. Result: {result}")
    return result

if __name__ == "__main__":
    seeds = [1, 2, 3, 4]
    print("[Main] Running MCMC chains in parallel...")

    with mp.Pool(processes=len(seeds)) as pool:
        results = pool.map(run_chain, seeds)

    print("[Main] All chains completed. Saving results...")

    # Save to Azure ML outputs directory
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/mcmc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[Main] Results saved to outputs/mcmc_results.json")