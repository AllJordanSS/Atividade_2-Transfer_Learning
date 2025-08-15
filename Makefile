# Variáveis
PYTHON = python
PIP = pip
DATA_DIR = Dataset-bunny_cat/classes
MODEL_PATH = checkpoints/vgg16_bunny_cat_model.h5

# Comandos principais
.PHONY: install train evaluate inference test clean setup

# Instalar dependências
install:
	$(PIP) install -r requirements.txt

# Configurar estrutura do projeto
setup:
	mkdir -p data/raw data/processed data/train data/validation data/test
	mkdir -p checkpoints logs results/plots results/evaluation
	mkdir -p notebooks tests

# Treinar modelo
train:
	$(PYTHON) scripts/train.py --epochs 10 --batch_size 128 --data_root $(DATA_DIR)

# Treinar com mais épocas
train-long:
	$(PYTHON) scripts/train.py --epochs 25 --batch_size 64 --data_root $(DATA_DIR)

# Avaliar modelo
evaluate:
	$(PYTHON) scripts/evaluate.py --model_path $(MODEL_PATH) --data_root $(DATA_DIR)

# Fazer inferência em uma imagem
inference:
	@echo "Usage: make inference IMAGE=path/to/image.jpg"
	@if [ -z "$(IMAGE)" ]; then echo "Por favor forneça o caminho da imagem: make inference IMAGE=path/to/image.jpg"; exit 1; fi
	$(PYTHON) scripts/inference.py --model_path $(MODEL_PATH) --image_path $(IMAGE)

# Migrar do código original
migrate:
	$(PYTHON) scripts/migrate_from_original.py --run_training

# Executar testes
test:
	$(PYTHON) -m pytest tests/ -v

# Executar testes com coverage
test-coverage:
	$(PYTHON) -m pytest tests/ --cov=models --cov=utils --cov=configs --cov-report=html

# Formatar código
format:
	black models/ utils/ configs/ scripts/ tests/
	flake8 models/ utils/ configs/ scripts/ tests/

# Limpar arquivos temporários
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete

# Limpar tudo (incluindo modelos e resultados)
clean-all: clean
	rm -rf checkpoints/*
	rm -rf logs/*
	rm -rf results/*
	rm -rf htmlcov/

# Visualizar estrutura do projeto
tree:
	tree -I '__pycache__|*.pyc|.git'

# Ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  install      - Instalar dependências"
	@echo "  setup        - Criar estrutura de pastas"
	@echo "  train        - Treinar modelo (10 épocas)"
	@echo "  train-long   - Treinar modelo (25 épocas)"
	@echo "  evaluate     - Avaliar modelo treinado"
	@echo "  inference    - Fazer inferência (use IMAGE=path/to/image.jpg)"
	@echo "  migrate      - Migrar do código original"
	@echo "  test         - Executar testes"
	@echo "  test-coverage- Executar testes com coverage"
	@echo "  format       - Formatar código"
	@echo "  clean        - Limpar arquivos temporários"
	@echo "  clean-all    - Limpar tudo"
	@echo "  tree         - Mostrar estrutura do projeto"
	@echo "  help         - Mostrar esta ajuda"