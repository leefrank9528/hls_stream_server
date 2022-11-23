import os
import sys
import traceback
import argparse
import typing as typ
import random
import time
from fractions import Fraction
import cv2
import numpy as np
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstVideo", "1.0")
from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo, GLib, GstVideoSink
import gstreamer.utils as utils

VIDEO_FORMAT = "RGB"
WIDTH, HEIGHT = 1920, 1080
FPS = Fraction(15)
GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(VIDEO_FORMAT)

print(os.environ)


def fraction_to_str(fraction: Fraction) -> str:
    """Converts fraction to str"""
    return '{}/{}'.format(fraction.numerator, fraction.denominator)


def parse_caps(pipeline: str) -> dict:
    """Parses appsrc's caps from pipeline string into a dict

    :param pipeline: "appsrc caps=video/x-raw,format=RGB,width=640,height=480 ! videoconvert ! autovideosink"

    Result Example:
        {
            "width": "640",
            "height": "480"
            "format": "RGB",
            "fps": "30/1",
            ...
        }
    """

    try:
        # typ.List[typ.Tuple[str, str]]
        caps = [prop for prop in pipeline.split(
            "!")[0].split(" ") if "caps" in prop][0]
        return dict([p.split('=') for p in caps.split(',') if "=" in p])
    except IndexError as err:
        return None


FPS_STR = fraction_to_str(FPS)
DEFAULT_CAPS = "video/x-raw,format={VIDEO_FORMAT},width={WIDTH},height={HEIGHT},framerate={FPS_STR}".format(**locals())

# Converts list of plugins to gst-launch string
# ['plugin_1', 'plugin_2', 'plugin_3'] => plugin_1 ! plugin_2 ! plugin_3
DEFAULT_PIPELINE = utils.to_gst_string([
    "appsrc emit-signals=True is-live=True caps={DEFAULT_CAPS}".format(**locals()),
    "queue",
    "videoconvert",
    "autovideosink"
])

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pipeline", required=False,
                default=DEFAULT_PIPELINE, help="Gstreamer pipeline without gst-launch")

ap.add_argument("-n", "--num_buffers", required=False,
                default=100, help="Num buffers to pass")

args = vars(ap.parse_args())

command = args["pipeline"]

args_caps = parse_caps(command)
NUM_BUFFERS = int(args['num_buffers'])

WIDTH = int(args_caps.get("width", WIDTH))
HEIGHT = int(args_caps.get("height", HEIGHT))
FPS = Fraction(args_caps.get("framerate", FPS))

GST_VIDEO_FORMAT = GstVideo.VideoFormat.from_string(
    args_caps.get("format", VIDEO_FORMAT))
CHANNELS = utils.get_num_channels(GST_VIDEO_FORMAT)
DTYPE = utils.get_np_dtype(GST_VIDEO_FORMAT)

FPS_STR = fraction_to_str(FPS)
CAPS = "video/x-raw,format={VIDEO_FORMAT},width={WIDTH},height={HEIGHT},framerate={FPS_STR}".format(**locals())

command = 'appsrc ! videoconvert ! vaapih264enc ! rtspclientsink location=rtsp://localhost:8554/ffstream01'

video_file_path = os.path.expanduser("~/Videos/27.mp4")
cap = cv2.VideoCapture(video_file_path)

with GstContext():  # create GstContext (hides MainLoop)
    print(f"command={command}")
    pipeline = GstPipeline(command)


    def on_pipeline_init(self):
        """Setup AppSrc element"""
        appsrc = self.get_by_cls(GstApp.AppSrc)[0]  # get AppSrc

        # instructs appsrc that we will be dealing with timed buffer
        appsrc.set_property("format", Gst.Format.TIME)

        # instructs appsrc to block pushing buffers until ones in queue are preprocessed
        # allows to avoid huge queue internal queue size in appsrc
        appsrc.set_property("block", True)

        # set input format (caps)
        appsrc.set_caps(Gst.Caps.from_string(CAPS))


    # override on_pipeline_init to set specific properties before launching pipeline
    pipeline._on_pipeline_init = on_pipeline_init.__get__(pipeline)

    try:
        pipeline.startup()
        appsrc = pipeline.get_by_cls(GstApp.AppSrc)[0]  # GstApp.AppSrc

        pts = 0  # buffers presentation timestamp
        duration = 10 ** 9 / (FPS.numerator / FPS.denominator)  # frame duration
        # for _ in range(NUM_BUFFERS):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # create random np.ndarray
            # array = np.random.randint(low=0, high=255,
            #                           size=(HEIGHT, WIDTH, CHANNELS), dtype=DTYPE)

            # convert np.ndarray to Gst.Buffer
            array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gst_buffer = utils.ndarray_to_gst_buffer(array)

            # set pts and duration to be able to record video, calculate fps
            pts += duration  # Increase pts by duration
            gst_buffer.pts = pts
            gst_buffer.duration = duration

            # emit <push-buffer> event with Gst.Buffer
            appsrc.emit("push-buffer", gst_buffer)

        # emit <end-of-stream> event
        appsrc.emit("end-of-stream")
        cap.release()

        while not pipeline.is_done:
            time.sleep(.1)
    except Exception as e:
        print("Error: ", e)
    finally:
        pipeline.shutdown()
