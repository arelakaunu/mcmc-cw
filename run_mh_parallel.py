import numpy as np
import multiprocessing as mp
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(processName)s] %(message)s")

def log_prior(theta):
    x, y = theta
    if -10 < x < 10 and -10 < y < 10:
        return 0.0
    return -np.inf

def log_likelihood(theta):
    x, y = theta
    mean1, mean2 = np.array([-2, -2]), np.array([3, 3])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    inv_cov = np.linalg.inv(cov)
    diff1, diff2 = theta - mean1, theta - mean2
    ll1 = -0.5 * diff1 @ inv_cov @ diff1
    ll2 = -0.5 * diff2 @ inv_cov @ diff2
    return np.logaddexp(ll1, ll2)

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

def metropolis_hastings(seed, steps=5000, burn_in=1000, proposal_std=0.5):
    np.random.seed(seed)
    current = np.random.uniform(-5, 5, size=2)
    samples = []
    for step in range(steps):
        proposal = current + np.random.normal(0, proposal_std, size=2)
        log_alpha = log_posterior(proposal) - log_posterior(current)
        if np.log(np.random.rand()) < log_alpha:
            current = proposal
        if step >= burn_in:
            samples.append(current.copy())
    samples = np.array(samples)
    return {
        "seed": seed,
        "samples": samples.tolist(),
        "mean": np.mean(samples, axis=0).tolist(),
        "std": np.std(samples, axis=0).tolist(),
    }

def save_results(results):
    os.makedirs("outputs", exist_ok=True)
    for i, res in enumerate(results):
        with open(f"outputs/mh_chain_{i+1}.json", "w") as f:
            json.dump(res, f, indent=2)
    summary = {
        "overall_mean": np.mean([r["mean"] for r in results], axis=0).tolist(),
        "overall_std": np.mean([r["std"] for r in results], axis=0).tolist()
    }
    with open("outputs/mh_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    seeds = [42, 1337, 2025, 9001]
    with mp.Pool(processes=len(seeds)) as pool:
        results = pool.map(metropolis_hastings, seeds)
    save_results(results)
    logging.info("✅ All MH chains complete.")