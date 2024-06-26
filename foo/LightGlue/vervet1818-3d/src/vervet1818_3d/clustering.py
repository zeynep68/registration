def get_linkage_matrix(model):
    import numpy as np
    
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix


def index_to_color(ix, labels, linkage_matrix):
    if ix < len(labels):
        return labels[ix]
    else:
        left_child_ix = int(linkage_matrix[ix - len(labels)][0])
        left_color = index_to_color(left_child_ix, labels, linkage_matrix)

        right_child_ix = int(linkage_matrix[ix - len(labels)][1])
        right_color = index_to_color(right_child_ix, labels, linkage_matrix)

        # assert left_color == right_color

        return left_color
