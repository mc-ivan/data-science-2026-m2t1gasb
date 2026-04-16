# Usa imagem base leve
FROM python:3.10-slim

# Define diretório de trabalho
WORKDIR /app

# Copia arquivos
COPY . .

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe porta
EXPOSE 5000

# Comando para rodar app
CMD ["gunicorn", "--reload", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]