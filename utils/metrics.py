import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import List

class MetricsCalculator:
    """Classe para calcular e visualizar métricas de avaliação"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
    
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula accuracy"""
        return accuracy_score(y_true, y_pred)
    
    def precision(self, y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
        """Calcula precision"""
        return precision_score(y_true, y_pred, average=average)
    
    def recall(self, y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
        """Calcula recall"""
        return recall_score(y_true, y_pred, average=average)
    
    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> float:
        """Calcula F1-score"""
        return f1_score(y_true, y_pred, average=average)
    
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula matriz de confusão"""
        return confusion_matrix(y_true, y_pred)
    
    def classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Gera relatório de classificação"""
        return classification_report(y_true, y_pred, target_names=self.class_names)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: str = None):
        """Plota matriz de confusão"""
        cm = self.confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusão')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 save_path: str = None):
        """Plota relatório de classificação como heatmap"""
        from sklearn.metrics import classification_report
        
        # Obter relatório como dicionário
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                     output_dict=True)
        
        # Extrair métricas por classe
        metrics_data = []
        for class_name in self.class_names:
            metrics_data.append([
                report[class_name]['precision'],
                report[class_name]['recall'],
                report[class_name]['f1-score']
            ])
        
        # Criar heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=['Precision', 'Recall', 'F1-Score'],
                   yticklabels=self.class_names, ax=ax)
        
        plt.title('Métricas por Classe')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()