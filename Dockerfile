FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Piper TTS binary
ARG PIPER_VERSION=2023.11.14-2
ARG TARGETARCH
RUN set -eux; \
        case "${TARGETARCH}" in \
            amd64) PIPER_ARCH="x86_64" ;; \
            arm64) PIPER_ARCH="aarch64" ;; \
            *) echo "Unsupported TARGETARCH: ${TARGETARCH}"; exit 1 ;; \
        esac; \
        wget -q "https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_${PIPER_ARCH}.tar.gz" -O /tmp/piper.tar.gz; \
        tar -xzf /tmp/piper.tar.gz -C /usr/local/bin --strip-components=1; \
        rm /tmp/piper.tar.gz

WORKDIR /app

COPY requirements.txt requirements-bench.txt ./
ARG INSTALL_BENCHMARK_DEPS=false
RUN set -eux; \
        pip install --no-cache-dir -r requirements.txt; \
        if [ "${INSTALL_BENCHMARK_DEPS}" = "true" ]; then \
            pip install --no-cache-dir -r requirements-bench.txt; \
        fi

COPY . .

# Directory for downloaded Piper voice models
RUN mkdir -p models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
