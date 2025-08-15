import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

class Visualizer:
    """Classe para visualização de resultados"""
    
    @staticmethod
    def plot_training_history(history, save_path: str = None):
        """Plota histórico de treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
        
        # Loss
        ax1.plot(history.history["loss"], label="Training Loss")
        ax1.plot(history.history["val_loss"], label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        
        # Accuracy
        ax2.plot(history.history["accuracy"], label="Training Accuracy")
        ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_prediction(img, probabilities: np.ndarray, class_names: List[str], 
                       true_label: str = None):
        """Visualiza predição de uma imagem"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Imagem
        ax1.imshow(img)
        ax1.set_title(f"Imagem{' - True: ' + true_label if true_label else ''}")
        ax1.axis('off')
        
        # Probabilidades
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        ax2.bar(class_names, probabilities)
        ax2.set_title(f"Predição: {predicted_class} ({confidence:.2%})")
        ax2.set_ylabel("Probabilidade")
        ax2.set_ylim(0, 1)
        
        # Destacar classe predita
        ax2.bar(predicted_class, confidence, color='red', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        return predicted_class, confidence