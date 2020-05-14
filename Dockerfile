FROM registry.redhat.io/ubi8/ubi:latest

LABEL io.openshift.tag=machineLearning

#COPY requirements.txt /tmp/reqirements.txt

RUN yum install python3 -y && \
pip3 install matplotlib numpy

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN pip3 install tensorflow
RUN pip3 install keras

#COPY train.py /tmp/train.py

CMD ["python3","/tmp/train.py"]
