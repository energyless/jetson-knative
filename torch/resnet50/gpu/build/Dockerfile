FROM dustynv/l4t-pytorch:r36.2.0

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

WORKDIR /app

COPY requirements.txt . 

RUN pip3 install -r requirements.txt

WORKDIR /app/src

COPY src .

# add signal to stop the container
STOPSIGNAL SIGINT

CMD "python3" "main.py"