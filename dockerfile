FROM python:3.12.8

WORKDIR /app

# pipのwarning削除用
ENV PIP_ROOT_USER_ACTION=ignore

COPY requirements.txt .

RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt

# モデルキャッシュ用のディレクトリを作成
RUN mkdir -p /root/.cache/fastembed_cache

# モデルをあらかじめダウンロードしておく
RUN python -c "from langchain_community.embeddings import FastEmbedEmbeddings; embeddings = FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large', cache_dir='/root/.cache/fastembed_cache')"

# ↑にて、既に必要モデルのダウンロードはすんでいるので,Hugging Faceのサーバへの問い合わせをしないように環境変数を設定
ENV HF_HUB_OFFLINE=1

# streamlitのサーバ用の環境変数（80番ポートで固定でサーバ立てるように、サーバ起動時にブラウザ起動しないように）
ENV STREAMLIT_SERVER_PORT=80
ENV STREAMLIT_SERVER_HEADLESS=true