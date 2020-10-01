from collections import OrderedDict
from itertools import compress
import numpy as np


def vars_and_indices(clauses):
    """
    Returns a tuple(OrderedDict(var:index),dict(index:var)
    In the first dict, the keys are ordered by the first occurrence of the DIMACS variables
    in the clauses (index is exactly the order of occurrence).
    The second dict is the reverse mapping (from index to DIMACS variable)
    :param clauses:list(list(int))
    :return:tuple(OrderedDict,dict)
    """
    var_set = OrderedDict()  # uniquely stores each variable in the original_clauses, preserving order of first occurrence
    for c in clauses:
        for lit in c:
            var_set[abs(lit)] = None

    # maps each variable to its index (order of occurrence)
    var_to_idx = {var: idx for idx, var in enumerate(var_set.keys())}

    # reverse mapping: index to variable
    idx_to_var = {idx: var for var, idx in var_to_idx.items()}
    return var_to_idx, idx_to_var


def num_vars(clauses):
    """
    Counts the actual number of variables in the original_clauses
    :param clauses:
    :return:
    """
    return len(vars_and_indices(clauses)[0])


def encode(clauses):
    """
    Encodes the given original_clauses as a factor graph and returns the dense
    adjacency matrix, with one node per variable and per clause
    Edges are between vars & original_clauses:
    - positive edge (+1) if var is asserted in clause
    - negative edge (-1) if var is negated in clause
    - no edge (0) if var is not present in clause
    Variable indexes in the original_clauses are according to their occurence in the original_clauses.
    E.g., if the original_clauses is: [[-5, 1], [2, -7, 5]] then the index of
    variables 5,1,2,7 become 0,1,2,3 respectively.
    :param clauses: list of original_clauses (each clause is a list of literals in DIMACS notation)
    :return: np.array with the adjacency matrix (#vars x #original_clauses), with +1/-1 for asserted/negated var in clause and 0
    if var not present in clause
    """

    # maps each variable to its index in the matrix
    var_to_idx, _ = vars_and_indices(clauses)
    adj = np.zeros((len(var_to_idx), len(clauses)))  # n x c adjacency matrix (n=#vars, c=#original_clauses)

    for c_num, clause in enumerate(clauses):
        for literal in clause:
            var_idx = var_to_idx[abs(literal)]
            adj[var_idx][c_num] = -1 if literal < 0 else 1

    return adj


def decode(adj_tensor, model):
    """
    Returns a list of original_clauses, given the adjacency tensor and model
    :param adj_tensor:
    :param model:
    :return:
    """
    raise NotImplementedError


def unit_propagation(formula, var, signal):
    """
    Performs unit propagation of a literal (=var*signal) in the original_clauses.
    That is, removes all original_clauses with literal and removes ~literal from the original_clauses it occurs
    :param formula:
    :param var: variable to be propagated
    :param signal: +1 if var is assigned True, -1 if False
    :return:
    """
    new_f = formula.copy()
    literal = (var+1)*signal  # brings to DIMACS notation

    # removes original_clauses with literal
    occurrences = [literal not in c for c in new_f.original_clauses]  # creates an array mask to keep original_clauses without literal
    new_f.original_clauses = list(compress(new_f.original_clauses, occurrences))  # filters the old list with the mask

    # removes occurrences of ~literal
    for c in new_f.original_clauses:
        # python triggers ValueError if the element is not on the list
        try:
            c.remove(-literal)
        except ValueError:
            pass  # ignore the error
    return new_f
