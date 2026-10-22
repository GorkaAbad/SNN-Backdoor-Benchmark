# One OCI image for Docker and Apptainer (docker:// images): the locked environment of this
# repository. torch and its CUDA libraries come from uv.lock, so the base is plain Python.
# Published on GHCR for linux/amd64 and linux/arm64 by .github/workflows/image.yml.
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

# No uv cache in the layers: the CUDA libraries alone take several GB.
ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/sbb/venv \
    PYTHONUNBUFFERED=1

WORKDIR /opt/sbb/src
# The dependencies first, so a code change does not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY benchmarks ./benchmarks
COPY replications ./replications
RUN uv sync --frozen --no-dev

ENV PATH=/opt/sbb/venv/bin:$PATH
# Run from a mounted working directory, with the cache and results there.
WORKDIR /work
CMD ["sbb", "--help"]
