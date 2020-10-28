import os
import subprocess
from collections import OrderedDict
from itertools import compress
import numpy as np
from copy import deepcopy
from pysat.formula import CNF


def create_sat_problem(sat_required, *args):
    """
    Creates a new instance of a SAT problem and returns the list of clauses.
    :param sat_required: is the problem instance required to be satisfiable?
    :param args: list of arguments as if cnfgen is being called from the command line
    :return:
    """
    # calls cnfgen to generate an instance
    subprocess.call(['cnfgen', '-q', '-o', 'tmp.cnf'] + [str(a) for a in args])
    if not sat_required:
        f = CNF(from_file='tmp.cnf')
        os.remove('tmp.cnf')
        return f.clauses

    else:  # the instance must be satisfiable, will check with minisat
        while True:
            try:
                subprocess.check_call(['minisat', 'tmp.cnf'], stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as ex:
                if ex.returncode == 10:  # instance is satisfiable, return it
                    f = CNF(from_file='tmp.cnf')
                    os.remove('tmp.cnf')
                    return f.clauses
                else:
                    # generates a new instance
                    subprocess.call(['cnfgen', '-q', '-o', 'tmp.cnf'] + args)



def vars_and_indices(clauses):
    """
    Returns a tuple(OrderedDict(var:index),dict(index:var)
    In the first dict, the keys are ordered by the first occurrence of the DIMACS variables
    in the clauses (index is exactly the order of occurrence).
    The second dict is the reverse mapping (from index to DIMACS variable)
    :param clauses:list(list(int))
    :return:OrderedDict,dict
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


def decode(adj_tensor, model):
    """
    Returns a list of original_clauses, given the adjacency tensor and model
    :param adj_tensor:
    :param model:
    :return:
    """
    raise NotImplementedError


def unit_propagation(clauses, var_idx, signal):
    """
    Performs unit propagation of a literal (=var*signal) in the original_clauses.
    That is, removes all original_clauses with literal and removes ~literal from the original_clauses it occurs
    :param clauses:
    :param var_idx: index of the variable to be propagated (starts with zero)
    :param signal: +1 if var is assigned True, -1 if False
    :return:
    """
    new_clauses = deepcopy(clauses)
    literal = (var_idx + 1) * signal  # brings to DIMACS notation

    # removes original_clauses with literal
    occurrences = [literal not in c for c in new_clauses]  # creates an array mask to keep original_clauses without literal
    new_clauses = list(compress(new_clauses, occurrences))  # filters the old list with the mask

    # removes occurrences of ~literal
    for c in new_clauses:
        # python triggers ValueError if the element is not on the list
        try:
            c.remove(-literal)
        except ValueError:
            pass  # ignore the error
    return new_clauses
