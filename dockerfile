FROM python:3.12.8

WORKDIR /app

# pipのwarning削除用
ENV PIP_ROOT_USER_ACTION=ignore

COPY requirements.txt .

RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt