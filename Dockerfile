FROM python:3.11.5-slim AS builder

WORKDIR /opt/pysetup

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 \
    --target /opt/pysetup/deps -r requirements.txt

FROM python:3.11.5-slim

WORKDIR /app

COPY --from=builder /opt/pysetup/deps /usr/local/lib/python3.11/site-packages

# Keep the runtime image self-contained so users can deploy directly from a
# published image without bind-mounting the whole repository.
COPY . .

ENV PATH="/usr/local/lib/python3.11/site-packages/bin:${PATH}"

EXPOSE 80
