FROM dustynv/l4t-pytorch:r36.2.0

WORKDIR /app

COPY requirements.txt . 

RUN pip3 install -r requirements.txt

WORKDIR /app

COPY . .

# add signal to stop the container
STOPSIGNAL SIGINT

CMD "python3" "main.py"