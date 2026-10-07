FROM python:3.11-slim

# uv installe les dépendances à partir de uv.lock, comme en local.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Les dépendances d'abord : cette couche n'est reconstruite que si uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

# Le CV, les ouvrages et la base ChromaDB ne sont pas dans l'image : ils sont déposés
# dans l'interface ou fournis par un volume monté sur /app/data.
RUN mkdir -p data/cv data/pdfs data/epubs data/chroma

EXPOSE 8501

# La clé OPENAI_API_KEY est fournie à l'exécution (docker run -e, fly secrets).
CMD ["streamlit", "run", "src/cv_assistant/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
