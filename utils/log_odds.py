import numpy as np
from collections import Counter


def compute_log_odds(tokens_class1, tokens_class0, alpha=0.01, min_freq=5):
    """
    Compute log-odds ratio with informative Dirichlet prior (Monroe et al., 2008)
    with variance normalization (z-score).

    Parameters
    ----------
    tokens_class1 : list
        List of tokens (or n-grams) for class 1 (e.g., empathy)

    tokens_class0 : list
        List of tokens (or n-grams) for class 0 (e.g., non-empathy)

    alpha : float
        Smoothing parameter for Dirichlet prior

    min_freq : int
        Minimum frequency threshold to filter rare tokens

    Returns
    -------
    z_scores : dict
        Dictionary {token: z-score}

    sorted_scores : list of tuples
        Sorted list of (token, score) from lowest to highest
    """

    # Count tokens
    counts_1 = Counter(tokens_class1)
    counts_0 = Counter(tokens_class0)

    vocab = set(counts_1.keys()).union(set(counts_0.keys()))

    total_1 = sum(counts_1.values())
    total_0 = sum(counts_0.values())

    # Background counts
    combined_counts = Counter(tokens_class1 + tokens_class0)
    background_total = sum(combined_counts.values())

    log_odds = {}
    variance = {}

    for token in vocab:

        count_1 = counts_1.get(token, 0)
        count_0 = counts_0.get(token, 0)
        count_bg = combined_counts.get(token, 0)

        # Apply informative prior
        alpha_w = count_bg * alpha

        num_1 = count_1 + alpha_w
        num_0 = count_0 + alpha_w

        den_1 = total_1 + alpha * background_total
        den_0 = total_0 + alpha * background_total

        # Log-odds
        delta = np.log(num_1 / (den_1 - num_1)) - np.log(num_0 / (den_0 - num_0))

        # Variance (for z-score)
        var = (1 / num_1) + (1 / num_0)

        log_odds[token] = delta
        variance[token] = var

    # Z-score normalization
    z_scores = {
        token: log_odds[token] / np.sqrt(variance[token])
        for token in vocab
    }

    # Filter rare tokens
    if min_freq is not None:
        z_scores = {
            token: score
            for token, score in z_scores.items()
            if combined_counts[token] >= min_freq
        }

    # Sort results
    sorted_scores = sorted(z_scores.items(), key=lambda x: x[1])

    return z_scores, sorted_scores


def get_top_k(sorted_scores, k=20):
    """
    Extract top-k tokens for both classes.

    Parameters
    ----------
    sorted_scores : list
        Output from compute_log_odds

    k : int
        Number of tokens to extract per class

    Returns
    -------
    top_class0 : list
        Lowest scores (class 0)

    top_class1 : list
        Highest scores (class 1)
    """

    top_class0 = sorted_scores[:k]
    top_class1 = sorted_scores[-k:]

    return top_class0, top_class1

def plot_log_odds(top_non, top_emp, title_non, title_emp):

    import matplotlib.pyplot as plt

    words_non = [" ".join(w[0]) if isinstance(w[0], tuple) else w[0] for w in top_non]
    scores_non = [w[1] for w in top_non]

    words_emp = [" ".join(w[0]) if isinstance(w[0], tuple) else w[0] for w in top_emp]
    scores_emp = [w[1] for w in top_emp]

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    axes[0].barh(words_non, scores_non, color='red')
    axes[0].set_title(title_non)
    axes[0].invert_yaxis()

    axes[1].barh(words_emp, scores_emp, color='blue')
    axes[1].set_title(title_emp)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()