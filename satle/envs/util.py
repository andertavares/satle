from itertools import compress

#TODO implement model as a tuple (-1,0,1) for false, free, sat, respectively

def encode(formula, model):
    """
    Returns a representation of the formula + partial solution
    :param formula:
    :param model:
    :return:
    """
    raise NotImplementedError


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
