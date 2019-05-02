FROM python:3.7
RUN apt-get update -y && apt-get install -y cmake
RUN pip install pipenv
WORKDIR /app
COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock
RUN pipenv install --deploy --system
COPY ./ ./

EXPOSE 8000
CMD ["gunicorn", "-w", "15", "-b", "0.0.0.0:8000", "-t", "120", "server:app", ]
