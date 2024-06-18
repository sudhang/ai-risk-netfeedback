import netin
from typing import Type, Callable, Union, Set
import numpy as np
import networkx as nx
import pandas as pd
import logging
import os

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.CRITICAL, format="%(asctime)s - %(message)s")


# monkey-patch netin.utils.constants.RANK_RANGE to be more granular
# netin.utils.constants.RANK_RANGE = np.arange(0.05, 1 + 0.05, 0.05)


class NetworkFeedbackLoopSimulator:
    """
    A class to simulate Networks that experience feedback loops.

    The idea is that the network is generated, and some ranking strategy (e.g. PageRank) is
    used to rank the nodes.  Then, some edges are rewired based on these rankings, i.e., a
    higher-ranking node is more likely to be chosen as the target node.

    This process is repeated for a number of iterations.  Does this worsen the already existing suppression of minority nodes?

    Attributes:
        netin_graph (Type[netin.Graph]): The graph on which to run the simulation.  It should already have had the .generate() method called on it.
        num_iterations (int): The number of iterations to run the simulation.
        rewire_pc (float): The percentage of edges to rewire in each iteration.
        metric (str): The metric to use to determine the fraction of minority nodes in the top-K nodes. One of netin.utils.constants.VALID_METRICS.
    """

    def __init__(
        self,
        netin_graph: Type[netin.Graph],
        num_iterations: int = 10,
        rewire_pc: float = None,
        rewire_num: int = None,
        metric: str = "pagerank",
    ):
        self._validate_constructor_arguments(
            netin_graph, num_iterations, rewire_pc, rewire_num, metric
        )

        self.graph = netin_graph
        self.num_iterations = num_iterations
        self.rewire_pc = rewire_pc
        self.rewire_num = rewire_num
        self.metric = metric
        self.minority_fractions = (
            []
        )  # This will eventually be a list of lists. Each inner list will contain the fraction of minority nodes in the top-K of nodes for each iteration.
        self.homophily_history_major = []
        self.homophily_history_minor = []

        self.logger = logging.getLogger(__name__)

    def _validate_constructor_arguments(
        self, netin_graph, num_iterations, rewire_pc, rewire_num, metric
    ):
        if netin_graph is None:
            raise ValueError("netin_graph cannot be None")
        if num_iterations < 1:
            raise ValueError("num_iterations must be greater than 0")
        if netin_graph.get_model_name() is None:
            raise ValueError(
                "Please call the .generate() method on the netin_graph first."
            )
        if metric not in netin.utils.constants.VALID_METRICS:
            raise ValueError(
                f"metric must be one of {netin.utils.constants.VALID_METRICS}"
            )

        if rewire_pc is not None and rewire_num is not None:
            raise ValueError("Only one of rewire_pc and rewire_num should be provided")
        if rewire_pc is None and rewire_num is None:
            raise ValueError("One of rewire_pc and rewire_num should be provided")
        if rewire_num is not None:
            if rewire_num < 0:
                raise ValueError("rewire_num must be greater than 0")
        if rewire_pc is not None:
            if rewire_pc < 0 or rewire_pc > 1:
                raise ValueError("rewire_pc must be between 0 and 1")

    def _examine_bias(self, metric: str, iteration: int):
        df = self.graph.get_node_metadata_as_dataframe()

        fraction_xs, fraction_ys = netin.stats.distributions.get_fraction_of_minority(
            df, self.metric
        )

        self.logger.debug(f"{fraction_xs=}, {fraction_ys=}")

        self.minority_fractions.append(fraction_ys)

        homophily_major, homophily_minor = netin.pah.infer_homophily(self.graph)
        self.homophily_history_major.append(homophily_major)
        self.homophily_history_minor.append(homophily_minor)

    def save_network(self, simname: str, iteration: int = None):
        directory = f"data/{simname}/graphs"
        if not os.path.exists(directory):
            os.makedirs(directory)
        nx.write_gexf(self.graph, f"{directory}/iteration_{iteration}.gexf")

    def run_simulation(self, simname: str = "default"):

        self.logger.info(f"Examining bias before rewiring")
        self._examine_bias(self.metric, iteration=-1)

        for i in range(self.num_iterations):
            self.logger.info(f"Doing iteration {i+1} of rewiring")

            rankings = self.graph.compute_node_stats(self.metric)

            # Update the graph based on the rankings
            self._rewire_edges(rankings)

            self.logger.info(f"Examining bias after iteration {i+1} of rewiring")
            self._examine_bias(self.metric, i)

            self.save_network(simname, iteration=i)

        # make a dataframe out of the above list of lists.
        df = pd.DataFrame(
            self.minority_fractions,
            columns=[f"{p:.2f}" for p in netin.utils.constants.RANK_RANGE],
        )
        # also add the homophily history to the dataframe
        df["homophily_major"] = self.homophily_history_major
        df["homophily_minor"] = self.homophily_history_minor
        df.index.name = "Iteration"

        return df

    def _rewire_edges(self, rankings):
        """Note: this method is private and modifies the graph in place."""
        # Now, we need to rewire the edges
        # the probability of a node to be chosen as the target node is proportional to its ranking

        edges = list(self.graph.edges())
        num_edges = len(edges)
        num_edges_to_rewire = (
            int(self.rewire_pc * num_edges)
            if self.rewire_pc is not None
            else self.rewire_num
        )
        num_nodes = len(self.graph.nodes())
        self.logger.info(f"{num_nodes=}, {num_edges=}, {num_edges_to_rewire=}")

        sampled_edges = np.random.choice(num_edges, num_edges_to_rewire, replace=False)

        for edge in sampled_edges:
            source, target = edges[edge]

            # potential targets can be all nodes except the source node or nodes that are already connected to the source node
            potential_targets = (
                set(self.graph.nodes()) - {source} - set(self.graph.neighbors(source))
            )
            # make sure that it an ordered set
            potential_targets = sorted(list(potential_targets))

            if len(potential_targets) > 0:
                # get the probabilities of choosing the target node
                probs, target_set = self._get_target_probabilities_for_rewiring(
                    rankings, source=source, target_set=potential_targets
                )

                # choose the target node
                new_target = np.random.choice(list(target_set), replace=False, p=probs)

                # rewire the edge
                self.graph.remove_edge(source, target)
                self.graph.add_edge(source, new_target)

                logging.info(f"rewired edge from {source} to {new_target}")

                original_target_class = self.graph.get_class_label(target)
                new_target_class = self.graph.get_class_label(new_target)
                if original_target_class != new_target_class:
                    logging.info(
                        f"Changed class of target node {target} from {original_target_class} to {new_target_class}"
                    )
                    logging.info(
                        f"Ranking was {rankings[target]}.  Is: {rankings[new_target]}"
                    )
                logging.info(
                    f"{probs=}, {target_set=}, {source=}, {target=}, {new_target=}"
                )

                if len(self.graph.edges()) != num_edges:
                    # investigate why this is happening
                    self.graph.add_edge(source, new_target)

    def _get_target_probabilities_for_rewiring(
        self,
        rankings: dict[int, float],
        source: Union[None, int],
        target_set: Union[None, Set[int]],
        special_targets: Union[None, object, iter] = None,
    ) -> tuple[np.array, set[int]]:
        scores = np.array(
            [rankings[node] + netin.utils.constants.EPSILON for node in target_set]
        )
        # Normalize the scores to get probabilities
        probs = scores / scores.sum()
        return probs, target_set


if __name__ == "__main__":
    G = netin.PAH(n=200, k=2, f_m=0.3, h_MM=0.7, h_mm=0.7)
    G.generate()

    sim = NetworkFeedbackLoopSimulator(
        netin_graph=G,
        num_iterations=20,
        rewire_num=10,
        metric="pagerank",
    )
    df = sim.run_simulation(simname="debug")
    print(df)

    # plot the column 0.1 of the dataframe
    plt.plot(df["0.10"])
    plt.plot(df["homophily_major"])
    plt.plot(df["homophily_minor"])
    plt.show()
