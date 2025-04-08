#!/bin/bash

docker compose kill
docker compose down
docker compose up -d
docker compose logs -f
