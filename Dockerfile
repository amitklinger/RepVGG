ARG base_image=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG timezone
FROM $base_image

ENV DEBIAN_FRONTEND=noninteractive, TZ=$timezone

RUN apt-get update && \
    apt-get -y --no-install-recommends install git build-essential python3-opencv wget vim sudo curl

ARG repo=https://github.com/amitklinger/RepVGG.git
ARG repvgg_branch=optimized_model

RUN git clone $repo --branch $repvgg_branch && \
    cd RepVGG && \
    pip install --upgrade pip

RUN pip install timm yacs termcolor loguru onnx onnxsim

ENV PYTHONPATH=/workspace/RepVGG
WORKDIR /workspace/RepVGG

RUN mkdir ckpts

RUN curl -L "https://drive.google.com/uc?export=download&id=13Gn8rq1PztoMEgK7rCOPMUYHjGzk-w11" -o ckpts/RepVGG-A0-train.pth
RUN curl -L "https://drive.google.com/uc?export=download&id=19lX6lNKSwiO5STCvvu2xRTKcPsSfWAO1" -o ckpts/RepVGG-A1-train.pth
RUN curl -L "https://drive.google.com/uc?export=download&id=1PvtYTOX4gd-1VHX8LoT7s6KIyfTKOf8G" -o ckpts/RepVGG-A2-train.pth

RUN curl -L "https://drive.google.com/uc?export=download&id=18g7YziprUky7cX6L6vMJ_874PP8tbtKx" -o ckpts/RepVGG-B0-train.pth
RUN curl -L "https://drive.google.com/uc?export=download&id=1VlCfXXiaJjNjzQBy3q7C3H2JcxoL0fms" -o ckpts/RepVGG-B1-train.pth
RUN curl -L "https://drive.google.com/uc?export=download&id=1cFgWJkmf9U1L1UmJsA8UT__kyd3xuY_y" -o ckpts/RepVGG-B2-train.pth

# RUN curl -L "https://drive.google.com/uc?export=download&id=1PL-m9n3g0CEPrSpf3KwWEOf9_ZG-Ux1Z" -o ckpts/RepVGG-B1g2-train.pth
# RUN curl -L "https://drive.google.com/uc?export=download&id=1WXxhyRDTgUjgkofRV1bLnwzTsFWRwZ0k" -o ckpts/RepVGG-B1g4-train.pth
# RUN curl -L "https://drive.google.com/uc?export=download&id=1LZ61o5XH6u1n3_tXIgKII7XqKoqqracI" -o ckpts/RepVGG-B2g4-train.pth

# RUN curl -L "https://drive.google.com/uc?export=download&id=1wBpq5317iPKk3-qblBHnx35bY_WumAlU" -o ckpts/RepVGG-B3-200epochs-train.pth
# RUN curl -L "https://drive.google.com/uc?export=download&id=1s7PxIP-oYB1a94_qzHyzfXAbbI24GYQ8" -o ckpts/RepVGG-B3g4-200epochs-train.pth
# RUN curl -L "https://drive.google.com/uc?export=download&id=16TcOlqKvTOf3M_l3ZWhxNPPc9MWtKZXY" -o ckpts/RepVGG-B2g4-200epochs-train.pth
