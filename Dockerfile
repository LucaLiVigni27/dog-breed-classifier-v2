# Small official Python image (Debian Linux, Python 3.12)
FROM python:3.12-slim

WORKDIR /app

# 1.) Pytorch first, CPU only: the slowest, biggest step, so it gets its own layer and it's
# reused from cache when only our code changes.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2.) Packages and demo's extra dependencies (Streamlit, paste button).
# README.md is needed because pyproject.toml declares it as the package readme.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[demo]"

# 3.) The app and its example photos.
COPY demo ./demo

# 4.) Streamlit default port, 0.0.0.0 makes it reachable from outside the container.
EXPOSE 8501
CMD ["streamlit", "run", "demo/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
