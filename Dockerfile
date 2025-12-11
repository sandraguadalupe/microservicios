# Imagen base
FROM python:3.10

# Crear carpeta de la app
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el proyecto completo
COPY . .

# Exponer el puerto (Railway lo asignará automáticamente)
EXPOSE 8000

# Comando para ejecutar FastAPI con Uvicorn
CMD ["python", "main.py"]
