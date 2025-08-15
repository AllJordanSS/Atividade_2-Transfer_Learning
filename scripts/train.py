import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.vgg16_transfer import VGG16Transfer
from utils.data_loader import DataLoader
from utils.visualization import Visualizer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Treinamento VGG16 Transfer Learning')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do batch')
    parser.add_argument('--data_root', type=str, default='data/raw/Dataset-bunny_cat/classes', 
                       help='Caminho para os dados')
    args = parser.parse_args()
    
    # Configuração
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_root=args.data_root
    )
    
    print("=== Configuração do Treinamento ===")
    print(f"Épocas: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Classes: {config.class_names}")
    print(f"Dados: {config.data_root}")
    
    # Carregar dados
    print("\n=== Carregando Dados ===")
    data_loader = DataLoader(config)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_loader.get_train_val_test_data()
    
    # Construir modelo
    print("\n=== Construindo Modelo ===")
    model = VGG16Transfer(num_classes=config.num_classes, freeze_features=True)
    model.build_model()
    model.compile_model(
        optimizer=config.optimizer,
        loss=config.loss_function,
        metrics=config.metrics
    )
    
    print("Resumo do modelo:")
    model.model.summary()
    
    # Treinar modelo
    print("\n=== Iniciando Treinamento ===")
    history = model.train(
        x_train, y_train,
        x_val, y_val,
        epochs=config.epochs,
        batch_size=config.batch_size
    )
    
    # Salvar modelo
    model_path = os.path.join(config.checkpoint_dir, 'vgg16_bunny_cat_model.h5')
    model.save_model(model_path)
    print(f"\nModelo salvo em: {model_path}")
    
    # Avaliar modelo
    print("\n=== Avaliação no Conjunto de Teste ===")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Visualizar resultados
    print("\n=== Visualizando Resultados ===")
    plot_path = os.path.join(config.results_dir, 'plots', 'training_history.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    visualizer = Visualizer()
    visualizer.plot_training_history(history, save_path=plot_path)
    
    print(f"Gráfico salvo em: {plot_path}")
    print("\n=== Treinamento Concluído ===")

if __name__ == "__main__":
    main()