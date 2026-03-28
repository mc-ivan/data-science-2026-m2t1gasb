![alt text](EMB-banner.png)

# Week 4 — Feature Vectors and Embeddings

Nesta semana aprofundamos o conceito de **representação de dados**, introduzindo **Feature Vectors** e **Embeddings**, fundamentais para o desempenho de modelos de Machine Learning e Deep Learning.

O foco está em entender como transformar dados brutos (como imagens) em representações numéricas mais informativas, permitindo que modelos aprendam padrões complexos de forma mais eficiente.

## Objetivo

Compreender como dados podem ser representados através de **vetores de características (Feature Vectors)** e **embeddings**, e avaliar como a qualidade dessa representação impacta diretamente o desempenho dos modelos de classificação.

## Conteúdos

- O que são **Feature Vectors**
- Representação de dados em alta dimensionalidade
- Limitações de usar dados brutos (pixels)
- Introdução a **Embeddings**
- Diferença entre:
  - Representação manual (raw data)
  - Representação aprendida (learned features)
- Conceito de **Transfer Learning**
- Redes pré-treinadas:
  - VGG16
- Extração de características (Feature Extraction)
- Redução de dimensionalidade (opcional):
  - Flatten
  - Pooling
- Impacto da representação no modelo
- Comparação entre abordagens:
  - Dados brutos vs embeddings
- Avaliação de modelos:
  - Accuracy
  - Tempo de treinamento

## Notebooks

Durante a semana implementamos um experimento prático para demonstrar a importância da representação dos dados:

1. **Feature Vectors and Embeddings — COVID-19 Radiography Dataset**  
   Implementação de dois classificadores:
   
   - **Classificador A:** treinado diretamente sobre os pixels das imagens  
   - **Classificador B:** utilizando **Transfer Learning com VGG16** para extração de embeddings  

   O objetivo é comparar o desempenho entre as duas abordagens e validar a hipótese de que **melhores representações levam a melhores resultados**.

   [![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/mc-ivan/data-science-2026-m2t1gasb/blob/main/week4/notebooks/Class4_FeatureVectorsAndEmbeddings.ipynb)

## Material da aula

Slides:  
[![View PDF](https://img.shields.io/badge/View-Slides_PDF-red?logo=adobeacrobatreader&logoColor=white)](content/4.FeatureVectorsAndEmbeddings.pdf)

## Autor

Eng. Ivan Mamani  

Responsável pelo desenvolvimento do conteúdo, material didático e notebooks desta semana.