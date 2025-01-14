# from joblib.parallel import method

import scTDC
import time
# import timeit
import numpy as np
from queue import Queue
# import os

# the next three lines should be implemented elsewhere, as attributes for instance
# NR_OF_MEASUREMENTS = 3    # number of measurements
# EXPOSURE_MS = 10        # exposure duration in milliseconds
# OUTPUT_TEXTFILE_NAME = "tmp_textfile.txt" # this file will be overwritten!

# define some constants to distinguish the type of element placed in the queue
QUEUE_DATA = 0
QUEUE_ENDOFMEAS = 1

device = scTDC.Device(autoinit=False)
# Value by which the raw data should be multiplied to get ps data arrays
digital_time_bin_resolution = 27.4


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
        self.data_length = 1000
        # open a BUFFERED_DATA_CALLBACKS pipe
        self.bufdatacb = BufDataCB(device.lib, device.dev_desc)

    @staticmethod
    def open_communication():
        # initialize TDC --- and check for error!
        retcode, errmsg = device.initialize()
        if retcode < 0:
            print("error during init:", retcode, errmsg)
            return False
        else:
            print("successfully initialized")
            return True

    @classmethod
    # define a closure that checks return codes for errors and does clean up
    def errorcheck(cls, retcode):
        if retcode < 0:
            print(device.lib.sc_get_err_msg(retcode))
            return True
        else:
            return False

    def set_exposure(self, ms_value):
        self.exposure_ms = ms_value

    def set_data_length(self, d_length):
        self.data_length = d_length


    def get_data(self):
        # start a first measurement
        retcode = self.bufdatacb.start_measurement(self.exposure_ms)
        if self.errorcheck(retcode):
            self.bufdatacb.close()
            device.deinitialize()

        eventtype, data = self.bufdatacb.queue.get()  # waits until element available
        # data = self.process_data(data, self.data_length)
        return data

    def close_communication(self):
        time.sleep(0.1)
        # clean up
        self.bufdatacb.close()  # closes the user callbacks pipe, method inherited from base class
        device.deinitialize()

