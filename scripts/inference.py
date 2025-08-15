import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.vgg16_transfer import VGG16Transfer
from utils.data_loader import DataLoader
from utils.visualization import Visualizer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Inferência com VGG16')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Caminho para o modelo salvo')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Caminho para a imagem de teste')
    args = parser.parse_args()
    
    # Configuração
    config = Config()
    
    # Carregar modelo
    print("Carregando modelo...")
    model = VGG16Transfer(num_classes=config.num_classes)
    model.load_model(args.model_path)
    
    # Carregar e processar imagem
    print(f"Processando imagem: {args.image_path}")
    data_loader = DataLoader(config)
    img, x = data_loader.get_image(args.image_path)
    
    # Fazer predição
    probabilities = model.predict(x)
    probabilities = probabilities[0]  # Primeira (e única) predição
    
    # Visualizar resultado
    visualizer = Visualizer()
    predicted_class, confidence = visualizer.plot_prediction(
        img, probabilities, config.class_names
    )
    
    print(f"\nResultado da Classificação:")
    print(f"Classe predita: {predicted_class}")
    print(f"Confiança: {confidence:.2%}")
    print(f"Probabilidades:")
    for i, class_name in enumerate(config.class_names):
        print(f"  {class_name}: {probabilities[i]:.2%}")

if __name__ == "__main__":
    main()