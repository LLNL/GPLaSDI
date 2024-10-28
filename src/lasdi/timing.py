"""

.. _NumPy docstring standard:
   https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

"""

from time import perf_counter

class Timer:
    """A light-weight timer class.
    """

    def __init__(self): 
        
        self.names = {}
        """:obj:`dict(str:int)`: Dictionary that maps job names to job indices."""

        self.calls = []
        """:obj:`list(int)`: List that stores the number of calls for each job."""

        self.times = []
        """:obj:`list(float)`: List that stores the total time for each job."""

        self.starts = []
        """
        :obj:`list(float)`: List that stores the start time for each job.
        If the job is not running, :obj:`None` is stored instead.
        """
        return
    
    def start(self, name):
        """Start a job named :obj:`name`.
        If the job is not listed, register the job in the job list.

        Args:
            name (:obj:`str`): Name of the job to be started.

        Note:
            The job must not have started before calling this method.

        Returns:
            Does not return a value.

        """

        if name not in self.names:
            self.names[name] = len(self.names)
            self.calls += [0]
            self.times += [0.0]
            self.starts += [None]
        
        idx = self.names[name]
        # check if the job is already being measured
        if (self.starts[idx] is not None):
            raise RuntimeError("Timer.start: %s timer is already ticking!" % name)
        self.starts[idx] = perf_counter()
        return
    
    def end(self, name):
        """End a job named :obj:`name`.
        Increase the number of calls and the runtime for the job.

        Args:
            name (:obj:`str`): Name of the job to be ended.

        Note:
            The job must have started before calling this method.

        Returns:
            Does not return a value.

        """
        assert(name in self.names)
        
        idx = self.names[name]
        # check if the job has started.
        if (self.starts[idx] is None):
            raise RuntimeError("Timer.end: %s start time is not measured yet!" % name)

        self.times[idx] += perf_counter() - self.starts[idx]
        self.calls[idx] += 1
        self.starts[idx] = None
        return
    
    def print(self):
        """Print the list of jobs and their number of calls, total time and time per each call.

        Returns:
            Does not return a value.
        """

        print("Function name\tCalls\tTotal time\tTime/call\n")
        for name, idx in self.names.items():
            print("%s\t%d\t%.3e\t%.3e\n" % (name, self.calls[idx], self.times[idx], self.times[idx] / self.calls[idx]))
        return
    
    def export(self):
        """Export the list of jobs and their number of calls and total time
        into a dictionary.

        Note:
            All jobs must be ended before calling this method.

        Returns:
            :obj:`dict` that contains "names", "calls", and "times" as keys
        """

        for start in self.starts:
            if (start is not None):
                raise RuntimeError('Timer.export: cannot export while Timer is still ticking!')

        param_dict = {}
        param_dict["names"] = self.names
        param_dict["calls"] = self.calls
        param_dict["times"] = self.times
        return param_dict
    
    def load(self, dict_):
        """Load the list of jobs and their number of calls and total time
        from a dictionary.

        Args:
            `dict_` (:obj:`dict`): Dictionary that contains the list of jobs and their calls and times.

        Note:
            :obj:`dict_['names']`, :obj:`dict_['calls']` and :obj:`dict_['times']` must have the same size.

        Returns:
            Does not return a value
        """

        self.names = dict_['names']
        self.calls = dict_['calls']
        self.times = dict_['times']

        assert(len(self.names) == len(self.calls))
        assert(len(self.names) == len(self.times))
        self.starts = [None] * len(self.names)
        return