FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --python 3.10 --system --deploy

COPY week4.py .

ENTRYPOINT [ "python", "week4.py" ]
