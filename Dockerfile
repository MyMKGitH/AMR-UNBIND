FROM mambaorg/micromamba:2.3.2

USER root
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n amr-unbind -f /tmp/environment.yml && \
    micromamba clean --all --yes

COPY . /opt/amr-unbind
WORKDIR /opt/amr-unbind

ENV PATH=/opt/conda/envs/amr-unbind/bin:$PATH
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
