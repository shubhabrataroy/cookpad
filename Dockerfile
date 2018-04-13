FROM python:2.7

ADD classifier_.py /

RUN pip install keras
RUN pip install numpy
RUN pip install matplotlib
RUN pip install scikit-learn

CMD ["python", "./classifier_.py"]

