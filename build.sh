#!/bin/bash

docker compose kill
docker compose down
docker compose build
