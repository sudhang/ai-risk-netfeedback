import netin
import itertools
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import concurrent.futures
import netfeedback as nf


def generate_network(
    network_size, minority_fraction, majority_homophily, minority_homophily
):
    G = netin.PAH(
        n=network_size,
        k=2,
        f_m=minority_fraction,
        h_MM=majority_homophily,
        h_mm=minority_homophily,
    )
    G.generate()
    return G


def run_single_experiment(params):
    (
        experiment_name,
        run_number,
        num_rewire_iterations,
        num_edges_to_rewire,
        metric,
        netin_graph,
    ) = params
    sim = nf.NetworkFeedbackLoopSimulator(
        netin_graph=netin_graph,
        num_iterations=num_rewire_iterations,
        rewire_num=num_edges_to_rewire,
        metric=metric,
    )
    df = sim.run_simulation(f"{experiment_name}/run_{run_number}")
    return df


def run_experiment(
    experiment_name,
    metric,
    network_size,
    rewire_num,
    num_iterations,
    majority_homophily,
    minority_homophily,
    minority_fraction,
    num_runs=10,
):
    netin_graph = generate_network(
        network_size, minority_fraction, majority_homophily, minority_homophily
    )

    params = [
        (
            experiment_name,
            run_number,
            num_iterations,
            rewire_num,
            metric,
            netin_graph,
        )
        for run_number in range(num_runs)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_experiment, params))

    average_df = pd.concat(results).groupby(level=0).mean()  # Improved averaging
    return average_df


def run_and_plot_experiment(
    experiment_name,
    metric,
    network_size,
    rewire_num,
    num_iterations,
    majority_homophily,
    minority_homophily,
    minority_fraction,
    num_runs=10,
):
    """Runs an experiment, saves results, and generates plots."""
    avg_df = run_experiment(
        experiment_name=experiment_name,
        metric=metric,
        network_size=network_size,
        rewire_num=rewire_num,
        num_iterations=num_iterations,
        majority_homophily=majority_homophily,
        minority_homophily=minority_homophily,
        minority_fraction=minority_fraction,
        num_runs=num_runs,
    )

    avg_df.to_csv(f"data/{experiment_name}/results.csv")

    create_plot(avg_df, experiment_name)
    return avg_df


def create_plot(df, experiment_name):
    fig1, ax = plt.subplots()
    ax.plot(df["0.10"], label="Minority Fraction")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Minority Fraction Trend for {experiment_name}")
    ax.legend()
    plt.savefig(f"data/{experiment_name}/minority_fraction.png")
    plt.close(fig1)

    fig2, ax = plt.subplots()
    ax.plot(df["homophily_major"], label="Homophily Majority")
    ax.plot(df["homophily_minor"], label="Homophily Minority")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Homophily")
    ax.set_title(f"Homophily Changes in {experiment_name}")
    ax.legend()
    plt.savefig(f"data/{experiment_name}/homophily.png")
    plt.close(fig2)


if __name__ == "__main__":

    run_and_plot_experiment(
        experiment_name="200NodesPagerankHighHomophily20EdgesRewired20Iterations",
        metric="pagerank",
        network_size=200,
        rewire_num=20,
        num_iterations=20,
        majority_homophily=0.9,
        minority_homophily=0.9,
        minority_fraction=0.3,
        num_runs=10,
    )
