#!/bin/bash

trap 'docker compose kill; docker compose down' INT

docker compose up -d
docker compose logs -f