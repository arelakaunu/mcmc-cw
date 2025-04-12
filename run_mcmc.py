import numpy as np
import emcee
import json
import os
import logging
from multiprocessing import Pool

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(processName)s] %(message)s")

# Prior: Uniform in a box
def log_prior(theta):
    x, y = theta
    if -10 < x < 10 and -10 < y < 10:
        return 0.0  # log(1)
    return -np.inf

# Likelihood: Mixture of Gaussians (e.g. multimodal target)
def log_likelihood(theta):
    x, y = theta
    mean1 = np.array([-2.0, -2.0])
    mean2 = np.array([3.0, 3.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    inv_cov = np.linalg.inv(cov)

    diff1 = np.array([x, y]) - mean1
    diff2 = np.array([x, y]) - mean2
    ll1 = -0.5 * diff1 @ inv_cov @ diff1
    ll2 = -0.5 * diff2 @ inv_cov @ diff2

    return np.logaddexp(ll1, ll2)  # log-sum-exp for numerical stability

# Full log-posterior
def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# Run a single MCMC sampler
def run_sampler(seed, n_walkers=40, n_steps=5000, burn_in=1000, thin=20, ndim=2):
    np.random.seed(seed)
    logging.info(f"Chain {seed}: Starting sampler")

    # Init walker positions around random spots within the prior
    p0 = np.random.uniform(low=-5, high=5, size=(n_walkers, ndim))

    # Set up emcee sampler
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)

    # Run burn-in
    p0, _, _ = sampler.run_mcmc(p0, burn_in, progress=True)
    sampler.reset()

    # Main sampling
    sampler.run_mcmc(p0, n_steps, progress=True)

    samples = sampler.get_chain(flat=True, thin=thin)
    autocorr_time = sampler.get_autocorr_time(tol=0)  # May raise warning if too short

    result = {
        "seed": seed,
        "mean": np.mean(samples, axis=0).tolist(),
        "std": np.std(samples, axis=0).tolist(),
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorrelation_time": autocorr_time.tolist(),
        "n_samples": len(samples),
    }

    logging.info(f"Chain {seed}: Completed")
    return result

if __name__ == "__main__":
    # Parallel seeds
    seeds = [42, 1337, 2025, 9001]

    logging.info("Running advanced MCMC with emcee in parallel...")

    with Pool(processes=len(seeds)) as pool:
        results = pool.map(run_sampler, seeds)

    # Save results
    os.makedirs("outputs", exist_ok=True)

    for i, result in enumerate(results):
        with open(f"outputs/emcee_chain_{i+1}_results.json", "w") as f:
            json.dump(result, f, indent=2)

    # Create summary across all chains
    summary = {
        "overall_mean": np.mean([r["mean"] for r in results], axis=0).tolist(),
        "overall_std": np.mean([r["std"] for r in results], axis=0).tolist(),
        "average_acceptance_rate": np.mean([r["acceptance_fraction"] for r in results]),
        "average_autocorrelation_time": np.mean([r["autocorrelation_time"] for r in results], axis=0).tolist(),
        "chains": len(results),
        "seeds": seeds
    }

    with open("outputs/emcee_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("✅ All chains complete. Results saved to 'outputs/'")