# Atividade_2-Transfer_Learning
Repositório com Projeto refinado da atividade 2 de Transfer Learning da rede VGG16.
Este projeto implementa um classificador de imagens usando transfer learning com VGG16 para distinguir entre coelhos (bunny) e gatos (cat).

## Estrutura do Projeto

```
vgg16_bunny_cat_classifier/
├── configs/          # Configurações do projeto
├── models/           # Implementação do modelo
├── utils/            # Utilitários (data loading, visualização)
├── scripts/          # Scripts de execução
├── data/             # Datasets
├── checkpoints/      # Modelos salvos
├── results/          # Resultados e plots
└── requirements.txt  # Dependências
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

### Treinamento
```bash
python scripts/train.py --epochs 10 --batch_size 128
```

### Inferência
```bash
python scripts/inference.py --model_path checkpoints/vgg16_bunny_cat_model.h5 --image_path path/to/image.jpg
```

## Dataset

Organize seu dataset na seguinte estrutura:
```
Dataset-bunny_cat/
└── classes/
    ├── bunny/
    │   ├── image1.jpg
    │   └── ...
    └── cat/
        ├── image1.jpg
        └── ...
```

## Resultados

O modelo utiliza transfer learning com VGG16 pré-treinada no ImageNet, congelando as camadas de features e retreinando apenas o classificador final.

## Características

- **Arquitetura**: VGG16 + classificador customizado
- **Pré-processamento**: Redimensionamento para 224x224, normalização ImageNet
- **Data Split**: 70% treino, 15% validação, 15% teste
- **Otimizador**: Adam
- **Loss**: Categorical Crossentropy