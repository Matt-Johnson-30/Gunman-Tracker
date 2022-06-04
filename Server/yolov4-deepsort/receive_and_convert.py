import sys
import traceback
import argparse
import typing as typ
import time
import attr

import numpy as np
import cv2
import datetime
import os

from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo
import gstreamer.utils as utils

################# GSTREAMER SETUP #################
# Converts list of plugins to gst-launch string
# ['plugin_1', 'plugin_2', 'plugin_3'] => plugin_1 ! plugin_2 ! plugin_3
DEFAULT_PIPELINE = utils.to_gst_string([
    "udpsrc port=5001 ", "application/x-srtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96, ssrc=(uint)3412089386, srtp-key=(buffer)012345678901234567890123456789012345678901234567890123456789, srtp-cipher=(string)aes-128-icm, srtp-auth=(string)hmac-sha1-80, srtcp-cipher=(string)aes-128-icm, srtcp-auth=(string)hmac-sha1-80, roc=(uint)0" , "srtpdec" , "rtph264depay", "h264parse", "decodebin", "videoconvert", "video/x-raw,format=(string)RGB", "queue", "appsink emit-signals=True"
])

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required=False, default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")

args = vars(ap.parse_args())
command = args["pipeline"]

COUNT = 0
################# GSTRAMER FUNCTIONS #################
def extract_buffer(sample: Gst.Sample) -> np.ndarray:
    """Extracts Gst.Buffer from Gst.Sample and converts to np.ndarray"""

    buffer = sample.get_buffer()  # Gst.Buffer

    caps_format = sample.get_caps().get_structure(0)  # Gst.Structure

    # GstVideo.VideoFormat
    video_format = GstVideo.VideoFormat.from_string(caps_format.get_value('format'))

    w, h = caps_format.get_value('width'), caps_format.get_value('height')
    c = utils.get_num_channels(video_format)

    buffer_size = buffer.get_size()
    shape = (h, w, c) if (h * w * c == buffer_size) else buffer_size
    array = np.ndarray(shape=shape, buffer=buffer.extract_dup(0, buffer_size), dtype=utils.get_np_dtype(video_format))

    return array


def on_buffer(sink: GstApp.AppSink, data: typ.Any) -> Gst.FlowReturn:
    """Callback on 'new-sample' signal"""
    global COUNT
    # Emit 'pull-sample' signal
    sample = sink.emit("pull-sample")  # Gst.Sample

    if isinstance(sample, Gst.Sample):
        array = extract_buffer(sample)
        ## WRITE TO FILE ##
        with open('./data/numpy_arrays/array{}.npy'.format(COUNT), 'wb') as f:
        	np.save(f, array)
        COUNT += 1
        if (COUNT >= 300):
             os.remove('./data/numpy_arrays/array{}.npy'.format(COUNT-300))
        return Gst.FlowReturn.OK

    return Gst.FlowReturn.ERROR        


################# MAIN #################
with GstContext():  # create GstContext (hides MainLoop)
    # create GstPipeline (hides Gst.parse_launch)
    with GstPipeline(command) as pipeline:
        appsink = pipeline.get_by_cls(GstApp.AppSink)[0]  # get AppSink
        # subscribe to <new-sample> signal
        appsink.connect("new-sample", on_buffer, None)
        while not pipeline.is_done:
            continue
