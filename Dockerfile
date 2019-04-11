FROM node:10 AS react-build
WORKDIR /usr/src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM kennethreitz/pipenv
WORKDIR /app
COPY --from=react-build /usr/src/frontend/build ./frontend/build
COPY clusterer/ ./clusterer/
COPY server.py ./

EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "server:app"]
