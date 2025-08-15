import os
import random
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.utils import to_categorical
from typing import Tuple, List, Dict

class DataLoader:
    """Classe para carregar e preprocessar dados"""
    
    def __init__(self, config):
        self.config = config
        self.data = []
        self.categories = []
        
    def get_image(self, path: str) -> Tuple:
        """Carrega e preprocessa uma imagem"""
        img = image.load_img(path, target_size=self.config.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x
    
    def load_data(self) -> List[Dict]:
        """Carrega todas as imagens do dataset"""
        root = self.config.data_root
        
        # Obter categorias
        self.categories = [x[0] for x in os.walk(root) if x[0]][1:]
        print(f"Categorias encontradas: {self.categories}")
        
        # Carregar imagens
        self.data = []
        for c, category in enumerate(self.categories):
            images = [
                os.path.join(dp, f) 
                for dp, dn, filenames in os.walk(category) 
                for f in filenames
                if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']
            ]
            
            print(f"Carregando {len(images)} imagens da categoria {category}")
            
            for img_path in images:
                try:
                    img, x = self.get_image(img_path)
                    self.data.append({'x': np.array(x[0]), 'y': c, 'path': img_path})
                except Exception as e:
                    print(f"Erro ao carregar {img_path}: {e}")
                    continue
        
        print(f"Total de imagens carregadas: {len(self.data)}")
        return self.data
    
    def split_data(self) -> Tuple:
        """Divide os dados em treino, validação e teste"""
        if not self.data:
            self.load_data()
            
        # Embaralhar dados
        random.shuffle(self.data)
        
        # Calcular índices de divisão
        total_size = len(self.data)
        idx_val = int(self.config.train_split * total_size)
        idx_test = int((self.config.train_split + self.config.val_split) * total_size)
        
        # Dividir dados
        train = self.data[:idx_val]
        val = self.data[idx_val:idx_test]
        test = self.data[idx_test:]
        
        print(f"Divisão dos dados:")
        print(f"  Treino: {len(train)} imagens")
        print(f"  Validação: {len(val)} imagens")
        print(f"  Teste: {len(test)} imagens")
        
        return train, val, test
    
    def prepare_arrays(self, data_split: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Converte lista de dados em arrays numpy"""
        x = np.array([item["x"] for item in data_split])
        y = np.array([item["y"] for item in data_split])
        
        # Normalizar dados
        x = x.astype('float32') / 255.0
        
        # Converter labels para one-hot encoding
        y = to_categorical(y, self.config.num_classes)
        
        return x, y
    
    def get_train_val_test_data(self) -> Tuple:
        """Retorna dados prontos para treinamento"""
        train, val, test = self.split_data()
        
        x_train, y_train = self.prepare_arrays(train)
        x_val, y_val = self.prepare_arrays(val)
        x_test, y_test = self.prepare_arrays(test)
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)