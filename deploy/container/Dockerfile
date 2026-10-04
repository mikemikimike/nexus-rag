# The system's image: the local-rag package and what it runs (the `confluence` command and
# LiteParse), nothing else. It does one thing, run one local-rag program; the MCP server is its
# default command, a Sync is the same image run with another command by a host timer.
# Built on the connected machine, from the repository root:
#   python3.12 -m local_rag.container save --out DIR     (or docker build -f deploy/container/Dockerfile .)
# or on the closed network from the mirrors, by `local_rag.container start --build`, which sets
# the build arguments below.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Where the build reads its packages from. The defaults are the public sources; the arguments
# end up in `docker history`, so give URLs without credentials.
#   APT_MIRROR            base URL of a Debian mirror that serves /debian and /debian-security
#   PIP_INDEX_URL         a PyPI index
#   NPM_CONFIG_REGISTRY   an npm registry
ARG APT_MIRROR=
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG NPM_CONFIG_REGISTRY=https://registry.npmjs.org/

# confluence-cli is a Node program, and the Confluence Corpus reads through it; the GitLab Corpus reads through git.
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i -E "s#https?://(deb|security)\.debian\.org#${APT_MIRROR%/}#g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends git nodejs npm \
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
