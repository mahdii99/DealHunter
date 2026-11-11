FROM python:3.12-slim
# Hugging Face Spaces runs the container as user 1000, so the app and its SQLite folder belong to that user
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
RUN mkdir -p data && chown user data
USER user
# 7860 = Hugging Face Spaces default port; Render/Koyeb pass $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
