FROM python:2
COPY . /src

RUN pip install numpy
RUN pip install h5py
RUN pip install pygame

RUN make -C /src/

CMD ["python", "/src/othello.py”]
