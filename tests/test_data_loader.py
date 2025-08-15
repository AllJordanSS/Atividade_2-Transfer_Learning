import unittest
import numpy as np
import tempfile
import os
import sys
from PIL import Image

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import DataLoader
from configs.config import Config

class TestDataLoader(unittest.TestCase):
    """Testes para a classe DataLoader"""
    
    def setUp(self):
        self.config = Config()
        # Criar diretório temporário para testes
        self.test_dir = tempfile.mkdtemp()
        
        # Criar estrutura de pastas de teste
        for class_name in ['bunny', 'cat']:
            class_dir = os.path.join(self.test_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Criar algumas imagens de teste
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
                img.save(os.path.join(class_dir, f'test_image_{i}.jpg'))
        
        self.config.data_root = self.test_dir
        self.data_loader = DataLoader(self.config)
    
    def tearDown(self):
        # Limpar arquivos de teste
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_load_data(self):
        """Testa carregamento de dados"""
        data = self.data_loader.load_data()
        
        self.assertEqual(len(data), 10)  # 5 imagens por classe, 2 classes
        
        # Verificar estrutura dos dados
        sample = data[0]
        self.assertIn('x', sample)
        self.assertIn('y', sample)
        self.assertIn('path', sample)
        
        # Verificar formato da imagem
        self.assertEqual(sample['x'].shape, (224, 224, 3))
    
    def test_split_data(self):
        """Testa divisão dos dados"""
        train, val, test = self.data_loader.split_data()
        
        total_size = len(train) + len(val) + len(test)
        self.assertEqual(total_size, 10)
        
        # Verificar proporções aproximadas
        train_ratio = len(train) / total_size
        val_ratio = len(val) / total_size
        
        self.assertAlmostEqual(train_ratio, 0.7, delta=0.2)
        self.assertAlmostEqual(val_ratio, 0.15, delta=0.2)
    
    def test_prepare_arrays(self):
        """Testa preparação dos arrays"""
        data = self.data_loader.load_data()
        x, y = self.data_loader.prepare_arrays(data[:5])
        
        self.assertEqual(x.shape[0], 5)
        self.assertEqual(x.shape[1:], (224, 224, 3))
        self.assertEqual(y.shape, (5, 2))  # One-hot encoding para 2 classes
        
        # Verificar normalização
        self.assertTrue(np.all(x >= 0) and np.all(x <= 1))

if __name__ == '__main__':
    unittest.main()