#!/bin/bash

ffmpeg -y -framerate 60 -pattern_type glob -i './IMG/center*.jpg' -c:v libx264 -pix_fmt yuv420p data.mp4
