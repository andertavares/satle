from collections import OrderedDict
from itertools import compress
import numpy as np


def var_dict(clauses):
    """
    Returns an OrderedDict(var:None), emulating an ordered set,
    where the keys are ordered by the first occurrence of the variables
    in the clauses
    :param clauses:list(list(int))
    :return:
    """
    var_set = OrderedDict()  # uniquely stores each variable in the formula, preserving order of first occurrence
    for c in clauses:
        for lit in c:
            var_set[abs(lit)] = None
    return var_set


def num_vars(clauses):
    """
    Counts the actual number of variables in the clauses
    :param clauses:
    :return:
    """
    return len(var_dict(clauses))


def encode(clauses):
    """
    Encodes the given formula as a factor graph and returns the dense
    adjacency matrix, with one node per variable and per clause
    Edges are between vars & clauses:
    - positive edge (+1) if var is asserted in clause
    - negative edge (-1) if var is negated in clause
    - no edge (0) if var is not present in clause
    Variable indexes in the clauses are according to their occurence in the formula.
    E.g., if the formula is: [[-5, 1], [2, -7, 5]] then the index of
    variables 5,1,2,7 become 0,1,2,3 respectively.
    :param clauses: list of clauses (each clause is a list of literals in DIMACS notation)
    :return: np.array with the adjacency matrix (#vars x #clauses), with +1/-1 for asserted/negated var in clause and 0
    if var not present in clause
    """
    variables = var_dict(clauses)

    # maps each variable to its index in the matrix
    var_to_idx = {var: idx for idx, var in enumerate(variables.keys())}
    adj = np.zeros((len(var_to_idx), len(clauses)))  # n x c adjacency matrix (n=#vars, c=#clauses)

    for c_num, clause in enumerate(clauses):
        for literal in clause:
            var_idx = var_to_idx[abs(literal)]
            adj[var_idx][c_num] = -1 if literal < 0 else 1

    return adj


def decode(adj_tensor, model):
    """
    Returns a list of clauses, given the adjacency tensor and model
    :param adj_tensor:
    :param model:
    :return:
    """
    raise NotImplementedError


def unit_propagation(formula, var, signal):
    """
    Performs unit propagation of a literal (=var*signal) in the formula.
    That is, removes all clauses with literal and removes ~literal from the clauses it occurs
    :param formula:
    :param var: variable to be propagated
    :param signal: +1 if var is assigned True, -1 if False
    :return:
    """
    new_f = formula.copy()
    literal = (var+1)*signal  # brings to DIMACS notation

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
