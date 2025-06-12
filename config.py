from dataclasses import dataclass
from typing import Optional

@dataclass
class HallAgentConfig:
    """Configuration for HallAgent4Rec"""
    # Model dimensions
    latent_dim: int = 64
    low_rank_dim: int = 16
    
    # Collaborative filtering regularization
    cf_reg_u: float = 0.01
    cf_reg_v: float = 0.01
    cf_learning_rate: float = 0.01
    cf_max_iterations: int = 100
    
    # Transfer learning
    transfer_reg: float = 0.1
    
    # Online learning parameters
    online_lr_theta: float = 0.01
    online_reg_theta: float = 0.1
    context_lr: float = 0.001
    bias_lr: float = 0.01
    
    # Hallucination mitigation
    replacement_alpha: float = 0.4
    
    # LLM parameters
    llm_temperature: float = 0.0
    llm_max_tokens: Optional[int] = None
    
    # Evaluation
    top_k: int = 10
    test_ratio: float = 0.2
    
    # Paths
    data_path: str = "./ml-100k/"
    model_save_path: str = "./models/"
    results_path: str = "./results/"
    
    # Random seed
    random_seed: int = 42