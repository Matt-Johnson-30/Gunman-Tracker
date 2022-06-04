#!/bin/bash

key="012345678901234567890123456789012345678901234567890123456789"
host="192.168.0.101"
port=5001
rtpcall="application/x-rtp, ssrc=(uint)3412089386"
bitrate=25000000

raspivid -t 0 -vf -hf -n -h 416 -w 600 -fps 10 -b $bitrate -o - | gst-launch-1.0 -q fdsrc ! h264parse ! \
rtph264pay config-interval=1 ! $rtpcall ! srtpenc key=$key ! tee name=t ! queue ! udpsink host=$host port=$port t. ! queue ! udpsink host=$host port=5002


#gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-h264,width=600,height=480,framerate=30/1 ! h264parse ! rtph264pay ! $rtpcall ! \
#srtpenc key=$key ! udpsink host=$host port=$port sync=false
