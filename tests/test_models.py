import unittest
import numpy as np
import tempfile
import os
import sys

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vgg16_transfer import VGG16Transfer
from configs.config import Config

class TestVGG16Transfer(unittest.TestCase):
    """Testes para a classe VGG16Transfer"""
    
    def setUp(self):
        self.config = Config()
        self.model = VGG16Transfer(num_classes=2)
    
    def test_build_model(self):
        """Testa a construção do modelo"""
        model = self.model.build_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape[-1], 2)  # 2 classes
        
    def test_compile_model(self):
        """Testa a compilação do modelo"""
        self.model.build_model()
        self.model.compile_model()
        
        # Verificar se o modelo foi compilado
        self.assertTrue(hasattr(self.model.model, 'optimizer'))
    
    def test_predict_shape(self):
        """Testa o formato da saída da predição"""
        self.model.build_model()
        self.model.compile_model()
        
        # Criar dados de teste
        x_test = np.random.random((1, 224, 224, 3))
        
        predictions = self.model.predict(x_test)
        
        self.assertEqual(predictions.shape, (1, 2))
        
    def test_save_load_model(self):
        """Testa salvamento e carregamento do modelo"""
        self.model.build_model()
        self.model.compile_model()
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            try:
                # Salvar modelo
                self.model.save_model(tmp.name)
                
                # Carregar modelo
                new_model = VGG16Transfer(num_classes=2)
                new_model.load_model(tmp.name)
                
                self.assertIsNotNone(new_model.model)
                
            finally:
                # Limpar arquivo temporário
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

if __name__ == '__main__':
    unittest.main()