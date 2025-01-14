import numpy as np
from fast_histogram import histogram1d
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from tdc_wrapper import BdcTdcWrapper

local_path = "O:\\lidyl\\atto\\Asterix\\NicolasG\\register_tdc_data_here"

digital_time_bin_resolution = 27.4

class DAQ_1DViewer_TDC(DAQ_Viewer_base):
    """ Instrument plugin class for a 1D viewer.
    
    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    TODO Complete the docstring of your plugin with:
        * The set of instruments that should be compatible with this instrument plugin.
        * With which instrument it has actually been tested.
        * The version of PyMoDAQ during the test.
        * The version of the operating system.
        * Installation instructions: what manufacturer’s drivers should be installed to make it run?

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.
         
    # TODO add your particular attributes here if any

    """
    params = comon_parameters+[
        # the nbr of measurement parameter will be implemented in an improved version of the plugin
        # {'title': 'Nbr of measurements:', 'name': 'meas_nbr', 'type': 'int', 'value': 1, 'min': 1},
        # {'title': 'Averaging Nbr:', 'name': 'Naverage', 'type': 'int', 'value': 1, 'min': 1},
        {'title': 'Line Settings:', 'name': 'line_settings', 'type': 'group', 'expanded': False, 'children': [
            {'title': 'CH1 Settings:', 'name': 'ch1_settings', 'type': 'group', 'expanded': True, 'children':
                [{'title': 'Enabled?:', 'name': 'enabled', 'type': 'bool', 'value': True}]},
            {'title': 'CH2 Settings:', 'name': 'ch2_settings', 'type': 'group', 'expanded': True, 'children':
                [{'title': 'Enabled?:', 'name': 'enabled', 'type': 'bool', 'value': False}]},
        ]},

        {'title': 'Acquisition:', 'name': 'acquisition', 'type': 'group', 'expanded': True, 'children': [
            {'title': 'Acq. type:', 'name': 'acq_type', 'type': 'list',
             'value': 'Histo', 'limits': ['Counting', 'Histo']},
            {'title': 'Exposure (ms):', 'name': 'exposure', 'type': 'int', 'value': 10, 'min': 1},
            {'title': 'Resolution (ps):', 'name': 'resolution', 'type': 'float', 'value': 1000, 'min': 0},
            {'title': 'Time window (ns):', 'name': 'window', 'type': 'float', 'value': 1, 'min': 0,
             'readonly': True, 'enabled': False, 'siPrefix': True},
            {'title': 'Nbins:', 'name': 'nbins', 'type': 'list', 'value': 1024,
             'limits': [1024 * (2 ** lencode) for lencode in range(6)]},
            {'title': 'Offset (ns):', 'name': 'offset', 'type': 'int', 'value': 0, 'max': 100000000, 'min': 0},
        ]},
        {'title': 'Data length:', 'name': 'data_length', 'type': 'int', 'value': 100, 'min': 1}
        ############
        ]

    def __init__(self, parent=None, params_state=None):

        super().__init__(parent, params_state) #initialize base class with common attributes and methods

        self.device = None
        self.x_axis = None
        self.controller = None
        self.datas = None  # list of numpy arrays, see set_acq_mode
        self.acq_done = False
        self.Nchannels = 0
        self.channels_enabled = {'CH1': {'enabled': True, 'index': 0}, 'CH2': {'enabled': False, 'index': 1}}
        self.h5saver = None
        self.timestamp_array = None
        self.ns_time_window = 1000
        self.ps_resolution = 1000
        self.nbins = int(self.ns_time_window / (self.ps_resolution / 1000))
        self.meas_nbr = 1
        self.mode = 'Histo'
        self.data_length = 100


    def ini_attributes(self):
        self.controller: BdcTdcWrapper


    def emit_log(self, string):
        self.emit_status(ThreadCommand('Update_Status', [string, 'log']))

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == 'resolution':    # we choose the window to remain the same --> the nbins changes
            self.ps_resolution = param.value()
            self.nbins = int(self.ns_time_window / (self.ps_resolution / 1000))
        if param.name() == 'nbins':         # we choose the window to remain the same --> the resolution changes
            self.nbins = param.value()
            self.ps_resolution = 1000 * self.ns_time_window / self.nbins
        if param.name() == 'window':        # we choose the nbins to remain the same --> the resolution changes
            self.ns_time_window = param.value()
            self.ps_resolution = 1000 * self.ns_time_window / self.nbins

        # if param.name() == 'meas_nbr':
        #     self.meas_nbr = param.value()

        if param.name() == 'exposure':
            self.controller.set_exposure(param.value())

        if param.name() == 'acq_type':
            self.mode = param.value()

        if param.name() == 'data_length':
            self.data_length = param.value()

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """

        self.ini_detector_init(slave_controller=controller)

        if self.is_master:
            self.controller = BdcTdcWrapper  #instantiate you driver with whatever arguments are needed
            self.controller.open_communication() # call eventual methods

        self.x_axis = self.get_xaxis()

        # # TODO for your custom plugin. Initialize viewers pannel with the future type of data
        # self.dte_signal_temp.emit(DataToExport(name='myplugin',
        #                                        data=[DataFromPlugins(name='Mock1',
        #                                                              data=[np.array([0., 0., ...]),
        #                                                                    np.array([0., 0., ...])],
        #                                                              dim='Data1D', labels=['Mock1', 'label2'],
        #                                                              axes=[self.x_axis])]))

        info = "TDC connected"
        initialized = True
        return info, initialized

    def get_xaxis(self):
        """
            Obtain the horizontal axis of the data.

            Returns
            -------
            1D numpy array
                Contains a vector of integer corresponding to the horizontal camera pixels.
        """
        if self.controller is not None:
            res = self.ps_resolution
            Nbins = self.nbins
            self.x_axis = Axis(data=np.linspace(0, (Nbins-1)*res, Nbins), label='Time', units='ps')
        else:
            raise(Exception('Controller not defined'))
        return self.x_axis

    def close(self):
        """Terminate the communication protocol"""
        self.controller.close_communication()  # when writing your own plugin replace this line

    @staticmethod
    def process_data(data):
        """
        transforms raw data into a ndarray containing only time of arrivals in ps

        data is a dictionary containing 1D arrays (and some integer values)

        returns a ps time array
        """
        actual_length = data["data_len"]  # true data length
        print("processing data chunk of length: {}".format(actual_length))
        # if actual_length > data_length:
        #     raise Exception("data length limit exceeded")
        # res = np.ndarray((data_length,), dtype=float)
        # res[:actual_length] = int(data["time"]) * digital_time_bin_resolution
        time_data = data["time"] * digital_time_bin_resolution
        return time_data

    @staticmethod
    def organise_0D_data(time_data, data_length):
        """
        Organises the data for the counting mode ==> 0D data viewer
        Args:
            time_data: contains the time of arrivals in ps (=result of process_data)
            data_length: size of the 0D data array to construct (must be constant for pymodaq)

        Returns:
            res: float np array filled w/ arrival times and zeros if there is fewer
            events than data_length
        """

        data_0D = np.zeros((data_length,))
        actual_length = len(time_data)
        data_0D[:actual_length] = time_data
        return data_0D

    @staticmethod
    def extract_histogram(raw_ps_times, Nbins, ps_time_window=None, channel=0):
        """
        Extract a histogram from raw data times in ps
        Args:
            raw_ps_times: (ndarray of int) electron arrival times
            Nbins: (int) the number of bins
            ps_time_window: (int) the maximum time value in ps
            channel: (int) marker of the specific channel (0 or 1) for channel 1 or 2

        Returns:
        ndarray: time of arrival histogram
        """
        hist = histogram1d(raw_ps_times, Nbins, (0, int(ps_time_window) - 1))
        return hist

    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """
        ## TODO for your custom plugin: you should choose EITHER the synchrone or the asynchrone version following

        ##synchrone version (blocking function)
        raw_data = self.controller.get_data()
        ps_time_data = self.process_data(raw_data)

        if self.mode == 'Histo':
            hist = self.extract_histogram(ps_time_data, Nbins=self.nbins)
            self.dte_signal.emit(DataToExport('myplugin',
                                              data=[DataFromPlugins(name='scTDC', data=hist,
                                                                    dim='Data1D',
                                                                    axes=[self.x_axis])]))

        if self.mode == 'Counting':
            raw_ps_events = self.organise_0D_data(ps_time_data, data_length=self.data_length)
            self.dte_signal.emit(DataToExport('myplugin',
                                              data=[DataFromPlugins(name='scTDC', data=raw_ps_events,
                                                                    dim='Data0D',)]))

        ##asynchrone version (non-blocking function with callback)
        # self.controller.your_method_to_start_a_grab_snap(self.callback)
        #########################################################


    def callback(self):
        """optional asynchrone method called when the detector has finished its acquisition of data"""
        data_tot = self.controller.your_method_to_get_data_from_buffer()
        self.dte_signal.emit(DataToExport('myplugin',
                                          data=[DataFromPlugins(name='Mock1', data=data_tot,
                                                                dim='Data1D', labels=['dat0', 'data1'])]))

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        self.controller.close_communication()  # when writing your own plugin replace this line
        self.emit_status(ThreadCommand('Update_Status', ['TDC Terminated']))
        ##############################
        return ''


if __name__ == '__main__':
    main(__file__)
