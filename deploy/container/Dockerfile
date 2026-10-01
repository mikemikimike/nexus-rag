# The system's image: the local-rag package and what it runs (the `confluence` command and
# LiteParse), nothing else. It does one thing, run one local-rag program; the MCP server is its
# default command, a Sync is the same image run with another command by a host timer.
# Built on the connected machine, from the repository root:
#   python3.12 -m local_rag.container save --out DIR     (or docker build -f deploy/container/Dockerfile .)
FROM python:3.12-slim

# confluence-cli is a Node program, and the Confluence Corpus reads through it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && npm install --global confluence-cli \
    && rm -rf /var/lib/apt/lists/* /root/.npm

WORKDIR /src
COPY requirements.lock pyproject.toml README.md ./
RUN pip install --no-cache-dir --only-binary :all: -r requirements.lock
COPY src ./src
RUN pip install --no-cache-dir --no-deps --no-build-isolation . && rm -rf /src

RUN useradd --system --create-home --home-dir /data --shell /usr/sbin/nologin ragsvc
USER ragsvc
WORKDIR /data
ENV HOME=/data

EXPOSE 8765
CMD ["local-rag-mcp"]
