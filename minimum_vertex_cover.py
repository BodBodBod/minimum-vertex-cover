from typing import Dict, List, Set, Tuple, Callable
from copy import deepcopy
import math
import random
import time


# approximation with Greedy Independent Set

def max_degree_node(graph: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    node = max(graph, key=lambda x: len(graph.get(x)))
    return node, graph[node]

def GIS_approximation(graph: Dict[str, List[str]]) -> Set[str]:

    graph_transform = deepcopy(graph)

    independent_set = set()

    while len(graph_transform) != 0:

        node, neighbours = max_degree_node(graph_transform)

        nodes_to_delete = [node] + neighbours

        independent_set.add(node)

        for v in nodes_to_delete:
            if v in graph_transform:
                del graph_transform[v]
        for v in graph_transform:
            graph_transform[v] = [x for x in graph_transform[v] if x not in nodes_to_delete]

    min_vertex_cover = set(graph.keys()) - independent_set
    return min_vertex_cover


# Simulated Annealing

# the number of edges that are not covered by the candidate solution

# pricing function is the number of uncovered edges
def pricing_function(graph: Dict[str, List[str]], candidate_solution: Set[str]) -> int:
    graph_copy = deepcopy(graph)

    diff_graph = {
        v: [u for u in graph_copy[v] if u not in candidate_solution]
        for v in graph_copy if v not in candidate_solution
    }

    cnt_uncovered_edges = sum(len(edges) for edges in diff_graph.values()) // 2
    return cnt_uncovered_edges

# maximum iteration count
# as (m - l - 1)^2,
# where m anb l is the number of nodes in the graph and in the current solution respectively
def make_max_iteration_cnt(graph: Dict[str, List[str]]) -> Callable[[Set[str]], int]:
    total_nodes = len(graph.keys())

    def max_iteration_cnt(current_solution: Set[str]) -> int:
        return (total_nodes - len(current_solution) - 1) ** 2

    return max_iteration_cnt


def is_vertex_cover(graph: Dict[str, List[str]], current_solution: Set[str]) -> bool:
    uncovered_edges = pricing_function(graph, current_solution)
    return uncovered_edges == 0


# random selection of the worst solution
def make_choice_between_solutions(graph: Dict[str, List[str]]) -> Set[str]:
    def choose_solution(current_solution: Set[str], sol: Set[str], T: float) -> Set[str]:
        current_cost = pricing_function(graph, current_solution)
        new_cost = pricing_function(graph, sol)
        delta = new_cost - current_cost

        if delta < 0:
            # better solution selection probability: 1
            return sol
        else:
            # worst solution selection probability: exp(-delta / T)
            acceptance_prob = math.exp(-delta / T) if T > 0 else 0
            if random.random() < acceptance_prob:
                return sol
            else:
                return current_solution

    return choose_solution

# random choice from set of nodes
def random_choice_from_set(s):
    return random.choice(list(s))

def graph_nodes(graph: Dict[str, List[str]]) -> Set[str]:
    return set(graph.keys())

def make_vertex_random_operation(graph: Dict[str, List[str]]) -> Callable[[Set[str], str], None]:
    graph_nodes_set = graph_nodes(graph)

    def vertex_random_operation(nodes: Set[str], op_type: str) -> None:
        if op_type == 'remove':
            # random deletion of a node
            nodes.remove(random_choice_from_set(nodes))
        if op_type == 'add':
            # random adding a node
            nodes.add(random_choice_from_set(graph_nodes_set - nodes))

    return vertex_random_operation

def simulated_annealing(
        graph: Dict[str, List[str]],
        initial_solution: Set[str] = None, # None for use GIS approximation as initial solution
        cutoff: int = 10, # cutoff time
        T: float = .05, # initial temperature
        T_coeff: float = .8 # temperature decreasing rate
) -> Set[str]:
    start_time = time.time()

    if initial_solution is None:
        # Greedy Independent Set approximation as initial solution
        initial_solution = GIS_approximation(graph)

    max_iteration_cnt = make_max_iteration_cnt(graph)  # maximum number of iterations for each approximation

    vertex_random_operation = make_vertex_random_operation(graph)  # randomly add and remove operations

    choose_solution = make_choice_between_solutions(graph)  # choosing between solutions

    sol = initial_solution
    updated_solution = initial_solution

    # print(f"Initial solution: {sol}")

    while time.time() - start_time < cutoff:
        T = T_coeff * T

        num_iter = 0  # number of iterations
        while num_iter <= max_iteration_cnt(sol):

            if is_vertex_cover(graph, sol):
                updated_solution = sol.copy()
                vertex_random_operation(sol, 'remove')
                # print(f"Update solution, new sol: {updated_solution}")

            current_solution = sol.copy()
            # print(f"Current solution: {current_solution}")

            vertex_random_operation(sol, 'remove')
            vertex_random_operation(sol, 'add')
            # print(f"Current solution: {current_solution}")
            # print(f"Sol: {sol}")

            sol = choose_solution(current_solution, sol, T)
            # print(f"New solution: {sol}")
            num_iter += 1

    return updated_solution