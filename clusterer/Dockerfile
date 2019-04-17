FROM python:3.7
RUN apt-get update -y && apt-get install -y cmake
RUN pip install pipenv
WORKDIR /app
COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock
RUN pipenv install --deploy --system
COPY ./ ./

EXPOSE 8000
CMD ["gunicorn", "--workers", "3", "-b", "0.0.0.0:8000", "server:app"]
