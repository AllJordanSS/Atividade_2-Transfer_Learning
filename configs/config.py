import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # Dados
    data_root: str = "Dataset-bunny_cat/classes"
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 128
    train_split: float = 0.7
    val_split: float = 0.15
    
    # Classes
    num_classes: int = 2
    class_names: list = None
    
    # Modelo
    pretrained: bool = True
    freeze_features: bool = True
    
    # Treinamento
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    metrics: list = None
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['bunny', 'cat']
        if self.metrics is None:
            self.metrics = ['accuracy']
            
        # Criar diretórios se não existirem
        for directory in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)