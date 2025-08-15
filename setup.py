from setuptools import setup, find_packages

setup(
    name="vgg16-bunny-cat-classifier",
    version="1.0.0",
    author="Seu Nome",
    author_email="seu.email@email.com",
    description="Classificador de imagens Bunny vs Cat usando VGG16 Transfer Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.8.0",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0",
        "Pillow>=8.3.0",
        "scikit-learn>=1.0.0",
        "seaborn>=0.11.0",
    ],
)