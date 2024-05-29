FROM python:3
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# 仮想環境の作成
RUN python -m venv /opt/venv

# 仮想環境を有効化し、パッケージをインストール
RUN /opt/venv/bin/pip install --upgrade pip
RUN /opt/venv/bin/pip install jupyterlab ipykernel

# 仮想環境のPythonをJupyterカーネルに登録
RUN /opt/venv/bin/python -m ipykernel install --name 'venv' --display-name "Python 3 (venv)"

# 仮想環境をデフォルトのPythonとして設定
ENV PATH="/opt/venv/bin:$PATH"

# Jupyter Labサーバーを起動
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
