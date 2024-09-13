# Hackathon SEMAC XXXIV
Desafio de Inteligência Artificial e Processamento de Linguagem Natural, com o tema `Análise de Sentimentos de Tweets`.

https://www.kaggle.com/competitions/hackathon-semac-xxxiv/overview

---
Utilizamos um modelo pré-treinado baseado na arquitetura RoBERTa, disponibilizado na plataforma Hugging Face pelo usuário Ibrahim-Alam [neste link](https://huggingface.co/Ibrahim-Alam/finetuning-roberta-base-on-tweet_sentiment_binary).


O arquivo `teste.py` foi utilizado para avaliar alguns modelos pré-treinados no dataset fornecido, com o intuito de escolher aquele que apresentasse melhor desempenho.

O arquivo `finetuning.py` realiza o treinamento fino do modelo pré-treinado escolhido no nosso conjunto de dados específico.

O arquivo `prediction.py` utiliza o modelo finalizado para realizar predições em cima dos dados de teste.
