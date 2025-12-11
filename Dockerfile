# Imagen base
FROM python:3.10

# Crear carpeta de trabajo
WORKDIR /app

# Copiar archivos
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto
EXPOSE 8000

# Ejecutar el microservicio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
