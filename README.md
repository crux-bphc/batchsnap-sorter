# BatchSnap Sorter

Identifies unique faces from corpus of photos and allows a person to retrieve all images containing their face. Clustering of images is done on server side by using facial feature embeddings, and the client facing portal is made in React.js.

## How it works

The initial processing happens on the server side.
1. The entire image corpus is processed by `clusterer/image_cluster.py` first to extract the necessary features from the faces.
2. Once the feature detection is complete, the embeddings are used to cluster all the detected faces into groups.

Normal Flow:
1. When a person uploads their photo, it is sent to the flask server.
2. Server runs feature detection on it. These features are matched with all clusters to find the closest cluster.
3. If a match is found, the server returns the links to all the images in that cluster.
4. Client side app retrieves the images from the server and zips them up for easy downloading.

## Getting Started
1. To run feature detection, use `pipenv run python image_cluster.py fd --path /path/to/images`.
2. To run clusterer, use `pipenv run python image_cluster.py cl`.
3. To run web application for normal flow, use `docker compose up`. See `docker-compose.yml`. The relevant environment variables are:
	- `IMAGES_DIR`: The location of the corpus on the server. Will be mapped to `/images` inside the docker container.
	- `RESULTS_FILE`: Path to pickle file generated from the clusterer, which contains the final output of the initial processing step.


## TODO
- [ ] Split `image_cluster.py` into feature detection and clusterer modules

## Contributors
- [Krut Patel](https://github.com/iamkroot)
- [Naman Arora](https://github.com/palindrome69)
- [Varun Parthasarathy](https://github.com/Var-ji)
