# Imagem base leve com Python
FROM python:3.10-slim

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia todos os arquivos do projeto
COPY . .

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe porta da aplicação
EXPOSE 5000

# Comando para rodar aplicação com Gunicorn (produção)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]