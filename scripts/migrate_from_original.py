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

def migrate_training():
    """Executa o treinamento usando a nova estrutura, equivalente ao código original"""
    
    print("=== Migrando do código original para estrutura organizada ===")
    
    # Configuração (equivalente às variáveis do código original)
    config = Config(
        data_root='Dataset-bunny_cat/classes',
        train_split=0.7,
        val_split=0.15,
        epochs=10,
        batch_size=128
    )
    
    print(f"Configurações aplicadas:")
    print(f"  Dataset: {config.data_root}")
    print(f"  Train split: {config.train_split}")
    print(f"  Val split: {config.val_split}")
    print(f"  Épocas: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    
    # 1. Carregar e preparar dados (substitui a seção de carregamento manual)
    print("\n1. Carregando dados...")
    data_loader = DataLoader(config)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_loader.get_train_val_test_data()
    
    print(f"Dados carregados:")
    print(f"  Treino: {x_train.shape}")
    print(f"  Validação: {x_val.shape}")
    print(f"  Teste: {x_test.shape}")
    
    # 2. Construir modelo VGG16 (substitui a criação manual do modelo)
    print("\n2. Construindo modelo VGG16...")
    model = VGG16Transfer(num_classes=config.num_classes, freeze_features=True)
    model.build_model()
    model.compile_model(
        optimizer=config.optimizer,
        loss=config.loss_function,
        metrics=config.metrics
    )
    
    print("Resumo do modelo:")
    model.model.summary()
    
    # 3. Treinar modelo (substitui model.fit())
    print("\n3. Iniciando treinamento...")
    history = model.train(
        x_train, y_train,
        x_val, y_val,
        epochs=config.epochs,
        batch_size=config.batch_size
    )
    
    # 4. Visualizar resultados de treinamento (substitui os plots manuais)
    print("\n4. Visualizando resultados do treinamento...")
    visualizer = Visualizer()
    plot_path = os.path.join(config.results_dir, 'plots', 'training_history.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    visualizer.plot_training_history(history, save_path=plot_path)
    
    # 5. Avaliar modelo (substitui model.evaluate())
    print("\n5. Avaliando modelo no conjunto de teste...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # 6. Salvar modelo
    model_path = os.path.join(config.checkpoint_dir, 'vgg16_bunny_cat_migrated.h5')
    model.save_model(model_path)
    print(f"\nModelo salvo em: {model_path}")
    
    # 7. Exemplo de inferência (substitui a seção de predição manual)
    print("\n6. Testando inferência...")
    
    # Usar uma imagem do conjunto de teste para demonstração
    test_idx = 0
    test_image = np.expand_dims(x_test[test_idx], axis=0)
    test_image_original = (x_test[test_idx] * 255).astype(np.uint8)  # Desnormalizar para visualização
    
    # Fazer predição
    probabilities = model.predict(test_image)
    probabilities = probabilities[0]
    
    # Visualizar resultado
    visualizer.plot_prediction(
        test_image_original, probabilities, config.class_names
    )
    
    # Mostrar probabilidades (equivalente ao print do código original)
    print("Probabilidades de classificação:", probabilities)
    predicted_class_index = np.argmax(probabilities)
    predicted_class = config.class_names[predicted_class_index]
    confidence = probabilities[predicted_class_index]
    
    print(f"A imagem foi classificada como: {predicted_class}")
    print(f"Com probabilidade de: {confidence:.2f}")
    
    # 8. Métricas detalhadas (melhoria em relação ao código original)
    print("\n7. Calculando métricas detalhadas...")
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    metrics_calc = MetricsCalculator(config.class_names)
    
    accuracy = metrics_calc.accuracy(y_true, y_pred)
    precision = metrics_calc.precision(y_true, y_pred)
    recall = metrics_calc.recall(y_true, y_pred)
    f1 = metrics_calc.f1_score(y_true, y_pred)
    
    print(f"\nMétricas detalhadas:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Salvar matriz de confusão
    cm_path = os.path.join(config.results_dir, 'evaluation', 'confusion_matrix.png')
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    metrics_calc.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    print(f"\n=== Migração Concluída ===")
    print(f"Estrutura organizada criada com sucesso!")
    print(f"Arquivos importantes:")
    print(f"  Modelo: {model_path}")
    print(f"  Gráfico de treinamento: {plot_path}")
    print(f"  Matriz de confusão: {cm_path}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Migração do código original')
    parser.add_argument('--run_training', action='store_true', 
                       help='Executar treinamento completo')
    args = parser.parse_args()
    
    if args.run_training:
        migrate_training()
    else:
        print("Para executar a migração completa, use: python migrate_from_original.py --run_training")
        print("\nEste script irá:")
        print("1. Carregar os dados usando a nova estrutura")
        print("2. Treinar o modelo VGG16")
        print("3. Salvar resultados organizados")
        print("4. Gerar visualizações")
        print("5. Calcular métricas detalhadas")

if __name__ == "__main__":
    main()
