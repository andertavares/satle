from itertools import compress
import numpy as np


def encode(num_vars, clauses):
    """
    Encodes the given formula as a factor graph and returns the dense
    adjacency matrix, with one node per variable and per clause
    Edges are between vars & clauses:
    - positive edge (+1) if var is asserted in clause
    - negative edge (-1) if var is negated in clause
    - no edge (0) if var is not present in clause
    :param clauses: list of clauses (each clause is a list of literals in DIMACS notation)
    :param num_vars: number of variables
    :return: np.array with the adjacency matrix (#vars x #clauses), with +1/-1 for asserted/negated var in clause and 0
    if var not present in clause
    """

    adj = np.zeros((num_vars, len(clauses)))  # n x c adjacency matrix (n=#vars, c=#clauses)

    for c_num, clause in enumerate(clauses):
        for literal in clause:
            var_index = abs(literal) - 1  # -1 to compensate that variables start at 1 in DIMACS
            adj[var_index][c_num] = -1 if literal < 0 else 1

    return adj


def decode(adj_tensor, model):
    """
    Returns a list of clauses, given the adjacency tensor and model
    :param adj_tensor:
    :param model:
    :return:
    """
    raise NotImplementedError


def unit_propagation(formula, literal):
    """
    Performs unit propagation of a literal in the formula.
    That is, removes all clauses with l and removes ~l from the clauses it occurs
    :param formula:
    :param literal:
    :return:
    """
    new_f = formula.copy()

    # removes clauses with literal
    occurrences = [literal not in c for c in new_f.clauses]  # creates an array mask to keep clauses without literal
    new_f.clauses = list(compress(new_f.clauses, occurrences))  # filters the old list with the mask

    # removes occurrences of ~literal
    for c in new_f.clauses:
        # python triggers ValueError if the element is not on the list
        try:
            c.remove(-literal)
        except ValueError:
            pass  # ignore the error
    return new_f
