FROM python:3.6

RUN pip install --upgrade pip

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . .

EXPOSE 8080

CMD [ "python", "app.py" ]


