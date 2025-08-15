import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from typing import List

class VGG16Transfer:
    """Classe para transfer learning com VGG16"""
    
    def __init__(self, num_classes: int, freeze_features: bool = True):
        self.num_classes = num_classes
        self.freeze_features = freeze_features
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """Constrói o modelo de transfer learning"""
        # Carregar VGG16 pré-treinada
        vgg = VGG16(weights='imagenet', include_top=True)
        
        # Referência à camada de entrada
        inp = vgg.input
        
        # Nova camada de classificação
        classification_layer = Dense(
            self.num_classes, 
            activation='softmax',
            name='predictions'
        )
        
        # Conectar nova camada à penúltima camada do VGG
        out = classification_layer(vgg.layers[-2].output)
        
        # Criar novo modelo
        self.model = Model(inp, out)
        
        # Congelar camadas se necessário
        if self.freeze_features:
            self._freeze_layers()
            
        return self.model
    
    def _freeze_layers(self):
        """Congela todas as camadas exceto a última"""
        for layer in self.model.layers[:-1]:
            layer.trainable = False
            
        # Garantir que a última camada seja treináve
        self.model.layers[-1].trainable = True
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', 
                     metrics=['accuracy']):
        """Compila o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi construído. Execute build_model() primeiro.")
            
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=128):
        """Treina o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi construído e compilado.")
            
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, x_test, y_test):
        """Avalia o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi construído.")
            
        return self.model.evaluate(x_test, y_test, verbose=0)
    
    def predict(self, x):
        """Faz predições"""
        if self.model is None:
            raise ValueError("Modelo não foi construído.")
            
        return self.model.predict(x)
    
    def save_model(self, filepath: str):
        """Salva o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi construído.")
            
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Carrega um modelo salvo"""
        self.model = keras.models.load_model(filepath)