import numpy as np

def demographic_parity(y_pred, sensitive_attr):
    """
    Measures demographic parity: P(positive|group1) = P(positive|group2)
    """
    unique = np.unique(sensitive_attr)
    rates = {}
    for u in unique:
        mask = sensitive_attr == u
        rates[u] = np.mean(y_pred[mask])
    return rates

def equalized_odds(y_true, y_pred, sensitive_attr):
    """
    Measures equalized odds: True positive/false positive rates equal across groups.
    """
    unique = np.unique(sensitive_attr)
    results = {}
    for u in unique:
        mask = sensitive_attr == u
        tpr = np.mean((y_pred[mask] == 1) & (y_true[mask] == 1)) / np.mean(y_true[mask] == 1) if np.mean(y_true[mask] == 1) > 0 else 0
        fpr = np.mean((y_pred[mask] == 1) & (y_true[mask] == 0)) / np.mean(y_true[mask] == 0) if np.mean(y_true[mask] == 0) > 0 else 0
        results[u] = {"TPR": tpr, "FPR": fpr}
    return results

def print_fairness_report(y_true, y_pred, sensitive_attr):
    print("=== Fairness Report ===")
    print("Demographic parity:", demographic_parity(y_pred, sensitive_attr))
    print("Equalized odds:", equalized_odds(y_true, y_pred, sensitive_attr))

if __name__ == "__main__":
    print("Fairness metrics module. Use print_fairness_report(y_true, y_pred, sensitive_attr).")