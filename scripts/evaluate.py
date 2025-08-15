import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.vgg16_transfer import VGG16Transfer
from utils.data_loader import DataLoader
from utils.visualization import Visualizer
from utils.metrics import MetricsCalculator
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Avaliação do modelo VGG16')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Caminho para o modelo salvo')
    parser.add_argument('--data_root', type=str, default='Dataset-bunny_cat/classes',
                       help='Caminho para os dados')
    args = parser.parse_args()
    
    # Configuração
    config = Config(data_root=args.data_root)
    
    # Carregar dados
    print("Carregando dados de teste...")
    data_loader = DataLoader(config)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_loader.get_train_val_test_data()
    
    # Carregar modelo
    print("Carregando modelo...")
    model = VGG16Transfer(num_classes=config.num_classes)
    model.load_model(args.model_path)
    
    # Fazer predições no conjunto de teste
    print("Fazendo predições...")
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calcular métricas
    print("Calculando métricas...")
    metrics_calc = MetricsCalculator(config.class_names)
    
    # Métricas básicas
    accuracy = metrics_calc.accuracy(y_true, y_pred)
    precision = metrics_calc.precision(y_true, y_pred)
    recall = metrics_calc.recall(y_true, y_pred)
    f1 = metrics_calc.f1_score(y_true, y_pred)
    
    print(f"\n=== Resultados da Avaliação ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Relatório de classificação
    print(f"\n=== Relatório de Classificação ===")
    classification_report = metrics_calc.classification_report(y_true, y_pred)
    print(classification_report)
    
    # Matriz de confusão
    print(f"\n=== Matriz de Confusão ===")
    cm = metrics_calc.confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Salvar visualizações
    results_dir = os.path.join(config.results_dir, 'evaluation')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot da matriz de confusão
    metrics_calc.plot_confusion_matrix(
        y_true, y_pred, 
        save_path=os.path.join(results_dir, 'confusion_matrix.png')
    )
    
    # Plot das métricas por classe
    metrics_calc.plot_classification_report(
        y_true, y_pred,
        save_path=os.path.join(results_dir, 'classification_report.png')
    )
    
    print(f"\nVisualizações salvas em: {results_dir}")

if __name__ == "__main__":
    main()