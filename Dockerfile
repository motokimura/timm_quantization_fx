FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN conda clean --all
RUN pip install timm==0.4.12 pandas matplotlib

ENV PYTHONPATH=/work
WORKDIR /work
