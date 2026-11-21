# Read-only API image: reports + aggregate results only (no data/, no models). See docs/azure.md.
FROM ghcr.io/astral-sh/uv:0.12.21 AS uv

FROM python:3.12.14-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock ./
# only the API group: fastapi + uvicorn (the modelling stack is not needed to serve reports)
RUN uv sync --locked --no-install-project --only-group api
COPY src/risikopegel/py.typed src/risikopegel/
COPY src/risikopegel/api src/risikopegel/api
COPY reports reports
COPY results results
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA PYTHONPATH=/app/src RISIKOPEGEL_ROOT=/app PATH=/opt/venv/bin:$PATH
RUN useradd --uid 10001 --no-create-home app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status != 200)"
CMD ["uvicorn", "risikopegel.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
