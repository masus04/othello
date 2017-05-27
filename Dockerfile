FROM python:2
COPY . /src

RUN cd src
RUN pip install numpy
RUN pip install h5py
RUN pip install pygame
RUN pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl 
RUN pip install torchvision

RUN make -C /src/

CMD ["python", "nohup python -u /src/othello.py > /src/othello_trainingdata.out &”]
