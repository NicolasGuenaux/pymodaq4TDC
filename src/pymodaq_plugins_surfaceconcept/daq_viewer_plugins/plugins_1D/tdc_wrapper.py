import scTDC
import time
import timeit
import numpy as np
from queue import Queue
import os

# the next three lines should be implemented elsewhere, as attributes for instance
# NR_OF_MEASUREMENTS = 3    # number of measurements
# EXPOSURE_MS = 10        # exposure duration in milliseconds
# OUTPUT_TEXTFILE_NAME = "tmp_textfile.txt" # this file will be overwritten!

# define some constants to distinguish the type of element placed in the queue
QUEUE_DATA = 0
QUEUE_ENDOFMEAS = 1

tdc = scTDC.Device(autoinit=False)
# initialize TDC --- and check for error!
retcode, errmsg = tdc.initialize()
if retcode < 0:
    print("error during init:", retcode, errmsg)
else:
    print("successfully initialized")

class BufDataCB(scTDC.buffered_data_callbacks_pipe):

    DATA_FIELD_SEL1 = \
        scTDC.SC_DATA_FIELD_TIME \
        | scTDC.SC_DATA_FIELD_CHANNEL \
        | scTDC.SC_DATA_FIELD_SUBDEVICE \
        | scTDC.SC_DATA_FIELD_START_COUNTER

    def __init__(self, lib, dev_desc,
                 data_field_selection=DATA_FIELD_SEL1,
                 max_buffered_data_len=500000,
                 dld_events=False):
        super().__init__(lib, dev_desc, data_field_selection,  # <-- mandatory!
                         max_buffered_data_len, dld_events)  # <-- mandatory!

        self.queue = Queue()
        self.end_of_meas = False


    def on_data(self, d):
        # make a dict that contains copies of numpy arrays in d ("deep copy")
        # start with an empty dict, insert basic values by simple assignment,
        # insert numpy arrays using the copy method of the source array
        dcopy = {}
        for k in d.keys():
            if isinstance(d[k], np.ndarray):
                dcopy[k] = d[k].copy()
            else:
                dcopy[k] = d[k]
        self.queue.put((QUEUE_DATA, dcopy))
        if self.end_of_meas:
            self.end_of_meas = False
            self.queue.put((QUEUE_ENDOFMEAS, None))

    def on_end_of_meas(self):
        self.end_of_meas = True
        # setting end_of_meas, we remember that the next on_data delivers the
        # remaining data of this measurement
        return True


class BdcTdcWrapper:

    def __init__(self):
        # the next 2 attributes could be part of an improvement of the plugin
        # but for now lets keep it simple and only adjust the exposure time
        # self.meas_nbr = 1
        # self.meas_remaining = self.meas_nbr
        self.exposure_ms = 10
        self.output_txtfile_name = "tmp_textfile.txt" # this file will be overwritten!

    def open_communication(self):
        # open a BUFFERED_DATA_CALLBACKS pipe
        self.bufdatacb = BufDataCB(tdc.lib, tdc.dev_desc)
        return

    def get_data(self):
        time = self.exposure_ms
        # start a first measurement
        retcode = self.bufdatacb.start_measurement(time)



