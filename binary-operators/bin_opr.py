import numpy as np

def _check_equal_length(emb1, emb2):
    """
    Check if two embeddings have the same length.

    Parameters:
    emb1 (np.array): The first node embedding.
    emb2 (np.array): The second node embedding.

    Raises:
    ValueError: If the lengths of emb1 and emb2 are not equal.
    """
    if len(emb1) != len(emb2):
        raise ValueError("The lengths of emb1 and emb2 must be equal.")

def average_opr(emb1, emb2):
    """
    Compute the average of two node embeddings.

    Parameters:
    emb1 (np.array): The embedding of the first node.
    emb2 (np.array): The embedding of the second node.

    Returns:
    np.array: The average embedding computed as (emb1 + emb2) / 2.

    Raises:
    ValueError: If the lengths of emb1 and emb2 are not equal.
    """
    _check_equal_length(emb1, emb2)
    return (emb1 + emb2) / 2

def hadamard_opr(emb1, emb2):
    """
    Compute the Hadamard product of two node embeddings.

    Parameters:
    emb1 (np.array): The embedding of the first node.
    emb2 (np.array): The embedding of the second node.

    Returns:
    np.array: The Hadamard product computed as emb1 * emb2.

    Raises:
    ValueError: If the lengths of emb1 and emb2 are not equal.
    """
    _check_equal_length(emb1, emb2)
    return emb1 * emb2

def weighted_l1_opr(emb1, emb2):
    """
    Compute the weighted L1 distance between two node embeddings.

    Parameters:
    emb1 (np.array): The embedding of the first node.
    emb2 (np.array): The embedding of the second node.

    Returns:
    np.array: The weighted L1 distance computed as |emb1 - emb2|.

    Raises:
    ValueError: If the lengths of emb1 and emb2 are not equal.
    """
    _check_equal_length(emb1, emb2)
    return np.abs(emb1 - emb2)

def weighted_l2_opr(emb1, emb2):
    """
    Compute the weighted L2 distance between two node embeddings.

    Parameters:
    emb1 (np.array): The embedding of the first node.
    emb2 (np.array): The embedding of the second node.

    Returns:
    np.array: The squared weighted L2 distance computed as |emb1 - emb2|^2.

    Raises:
    ValueError: If the lengths of emb1 and emb2 are not equal.
    """
    _check_equal_length(emb1, emb2)
    return np.power(np.abs(emb1 - emb2), 2)

if __name__ == "__main__":
    emb_x = np.random.randint(1, 100, size=5)
    emb_y = np.random.randint(1, 100, size=5)

    print('X ->', emb_x)
    print('Y ->', emb_y)

    print('Average ->', average_opr(emb_x, emb_y))
    print('Hadamard ->', hadamard_opr(emb_x, emb_y))
    print('Weighted-L1 ->', weighted_l1_opr(emb_x, emb_y))
    print('Weighted-L2 ->', weighted_l2_opr(emb_x, emb_y))