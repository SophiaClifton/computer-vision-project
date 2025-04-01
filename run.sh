#!/bin/bash

docker compose kill
docker compose down
docker compose up -d
docker compose exec -it cuda_opencv bash -c "python demo/app.py"
