from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"

    # Rollout
    group_size: int = 8          # G completions per problem
    temperature: float = 0.6
    max_new_tokens: int = 2000
    max_seq_len: int = 2048

    # Training
    learning_rate: float = 1e-6
    batch_size: int = 4          # problems per batch
    max_grad_norm: float = 1.0
    n_epochs: int = 3

    # Loss
    clip_eps: float = 0.2
    kl_coef: float = 0.01

    # Logging / saving
    log_every: int = 10
    save_every: int = 100
    output_dir: str = "checkpoints"
