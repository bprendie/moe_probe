FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install the necessary transformer and UI libraries
RUN pip install --no-cache-dir transformers accelerate rich sentencepiece

# Keep the container alive so we can attach an interactive shell
CMD ["/bin/bash", "-c", "tail -f /dev/null"]
