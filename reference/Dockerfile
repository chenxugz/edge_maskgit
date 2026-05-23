# Reproducible Milestone 1 reference oracle (PyTorch).
# Build:  docker build -t maskgit-ref reference/
# Run:    docker run --rm -v "$PWD/reference:/work" maskgit-ref \
#             python run_reference.py --class-id 207 --seed 42 --steps 8 -o output.png
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /work/requirements.txt
# CPU-only torch wheel keeps the image small and host-independent.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 \
    && pip install --no-cache-dir \
        numpy==2.4.6 pillow==12.2.0 einops==0.8.2 ml_collections==1.1.0 \
        tqdm==4.67.3 requests==2.34.2 matplotlib==3.10.9 pytest==9.0.3

COPY . /work

# Checkpoints are large (~880 MB) and fetched at runtime, not baked into the image.
CMD ["bash", "-lc", "./download_checkpoints.sh && python run_reference.py --class-id 207 --seed 42 --steps 8 -o output.png"]
