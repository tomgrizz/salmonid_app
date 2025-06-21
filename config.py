import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "PekingU/rtdetr_v2_r50vd"
    image_size: int = 480
    
    # Training settings
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 0.01
    
    # Data settings
    dataloader_num_workers: int = 2
    coco_folder: str = "/path/to/your/coco/dataset"  # Update this path
    
    # Output settings
    output_dir: str = "rt-detr_finetuned"
    save_total_limit: int = 2
    
    # Hardware settings
    fp16: bool = False
    cuda_device: str = "0"  # Set to "0" for first GPU, "1" for second, etc.
    
    def __post_init__(self):
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device

# Create default config
default_config = TrainingConfig() 