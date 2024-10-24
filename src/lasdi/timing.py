# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from time import perf_counter



# -------------------------------------------------------------------------------------------------
# Timer class
# -------------------------------------------------------------------------------------------------

class Timer:
    def __init__(self):
        # Set instance variables
        self.names      : dict  = {}    # Dictionary of named timers. key = name, value = index 
        self.calls      : list  = []    # k'th element = # of times we have called the k'th timer
        self.times      : list  = []    # k'th element = total time recorded by the k'th timer
        self.starts     : list  = []    # k'th element = start time for the k'th timer (if running)
        return
        
    

    def start(self, name : str) -> None:
        """
        Starts a specific timer. The user must specify the name of the timer they want to start. 
        The specified timer can not already be running!

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        name: A string specifying the name of the timer you want to start.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # If the timer does not already exist, initialize it
        if name not in self.names:
            # If this is the k'th timer that we have added to self (which would be the case if 
            # self.names has k keys), then we store that information in the "names" dictionary. 
            # We then store other information about that timer (such as its current value) in the
            # k'th element of the calls/times/starts lists.
            self.names[name] = len(self.names)
            self.calls      += [0]
            self.times      += [0.0]
            self.starts     += [None]
        
        # Fetch the current timer's number.
        idx : int = self.names[name]

        # Make sure the current timer has not already started. If so, it is already running and we
        # need to raise an exception.
        if (self.starts[idx] is not None):
            raise RuntimeError("Timer.start: %s timer is already ticking!" % name)
        
        # Set the current timer's start element to the time when we started this timer.
        self.starts[idx] = perf_counter()

        # All done!
        return
    


    def end(self, name : str) -> None:
        """
        Stops a specific timer. The user must specify the name of the timer they want to stop.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        name: A string specifying the name of the timer you want to stop.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Make sure the requested timer actually exists.
        assert(name in self.names)
        
        # Fetch the requested timers' index.
        idx = self.names[name]
        
        # Make sure the requested timer is actually running.
        if (self.starts[idx] is None):
            raise RuntimeError("Timer.end: %s start time is not measured yet!" % name)

        # Record the time since the current timer started, add it to the running total for this 
        # timer. Also increment the number of calls to this timer + reset it's start value to 
        # None (so we can start it again).
        self.times[idx] += perf_counter() - self.starts[idx]
        self.calls[idx] += 1
        self.starts[idx] = None

        # All done!
        return
    


    def print(self) -> None:
        """
        This function reports information on every timer in self. It has no arguments and returns 
        nothing.
        """
        
        # Header
        print("Function name\tCalls\tTotal time\tTime/call\n")

        # Cycle through timers.
        for name, idx in self.names.items():
            print("%s\t%d\t%.3e\t%.3e\n" % (name, self.calls[idx], self.times[idx], self.times[idx] / self.calls[idx]))
        
        # All done!
        return
    


    def export(self) -> dict:
        """
        This function extracts the names, calls, and times attributes of self, stores them in a 
        dictionary, and then returns that dictionary. If you have another dictionary object, 
        you can passed the returned dictionary to that object's load method to make that object 
        into an identical copy of self.
        
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        None!


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        A dictionary housing the names, calls, and times attributes of self. The returned 
        dictionary has three keys:
            - names
            - calls
            - times
        
        names is a dictionary with string keys whose corresponding values are integer indexes. If a 
        particular timer was the k'th one added to self, then it's value in names will be k. 

        calls is a list whose k'th element specifies how many times the k'th timer was stopped.

        times is a list whose k'th element specifies the total time recorded on the k'th timer.
        """

        # Make sure that no timers are currently running.
        for start in self.starts:
            if (start is not None):
                raise RuntimeError('Timer.export: cannot export while Timer is still ticking!')

        # Set up a dictionary to house the timer information.
        param_dict : dict = {}

        # Store names, calls, and timers (but not starts... we don't need that) in the dictionary.
        param_dict["names"] = self.names
        param_dict["calls"] = self.calls
        param_dict["times"] = self.times

        # All done!
        return param_dict
 


    def load(self, dict_ : dict) -> None:
        """
        This function de-serializes a timer object, making self into an identical copy of a 
        previously serialized timer object. Specifically, we replace self's names, calls, and 
        times attributes using those in the passed dict_. We use this function to restore a 
        timer object's state after loading from a checkpoint.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        dict_: This should be a dictionary with three keys:
            - names
            - calls
            - times
        the corresponding values should be the names, calls, and times attributes of another timer
        object, respectively. We replace self's attributes with those the values in dict_. dict_ 
        should be the dictionary returned by calling export on a timer object.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """
        self.names = dict_['names']
        self.calls = dict_['calls']
        self.times = dict_['times']

        assert(len(self.names) == len(self.calls))
        assert(len(self.names) == len(self.times))
        self.starts = [None] * len(self.names)
        return