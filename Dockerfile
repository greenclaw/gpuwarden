# gpuwarden in a container: cron lives INSIDE the image, so the host gets no new system daemon.
# The tool only makes HTTPS calls to RunPod — no Docker socket, no privileged mode, no docker-in-docker.
FROM python:3.12-slim

ARG RUNPODCTL_VERSION=v2.7.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash curl ca-certificates cron tini tzdata \
    && curl -fsSL -o /usr/local/bin/runpodctl \
        "https://github.com/runpod/runpodctl/releases/download/${RUNPODCTL_VERSION}/runpodctl-linux-amd64" \
    && chmod +x /usr/local/bin/runpodctl \
    && rm -rf /var/lib/apt/lists/*
# curl stays: cloud.sh needs it at runtime (GraphQL balance, pod health polling)

COPY . /src
RUN pip install --no-cache-dir /src && rm -rf /src

# Config and keys are mounted, never baked in.
ENV GPUWARDEN_CONF_DIR=/config
VOLUME ["/config", "/models"]

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# tini reaps cron's children; cron runs in the foreground as PID-managed process.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
