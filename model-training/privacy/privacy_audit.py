from utils.audit_logging import log_event

def log_dp_training(model_name, version, epsilon, delta, user="system"):
    log_event("dp_training", user=user, extra=f"{model_name}:{version} (ε={epsilon}, δ={delta})")

def log_federated_round(model_name, round_num, clients, user="system"):
    log_event("federated_round", user=user, extra=f"{model_name} round={round_num} clients={clients}")

def log_secure_agg(model_name, round_num, user="system"):
    log_event("secure_aggregation", user=user, extra=f"{model_name} round={round_num}")

if __name__ == "__main__":
    log_dp_training("melodygen", "1.0.3", 2.1, 1e-5, user="privacy-bot")
    log_federated_round("melodygen", 2, ["client1", "client2"], user="privacy-bot")
    log_secure_agg("melodygen", 2, user="privacy-bot")