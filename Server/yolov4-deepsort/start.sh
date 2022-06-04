#!/bin/bash

python "object_tracker_from_file.py" &
echo "Loading Tensorflow..."
sleep 16
python "receive_and_convert.py" &

