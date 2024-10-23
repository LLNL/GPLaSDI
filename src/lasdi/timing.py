from time import perf_counter

class Timer:
    def __init__(self):
        self.names = {}
        self.calls = []
        self.times = []
        self.starts = []
        return
    
    def start(self, name):
        if name not in self.names:
            self.names[name] = len(self.names)
            self.calls += [0]
            self.times += [0.0]
            self.starts += [None]
        
        idx = self.names[name]
        if (self.starts[idx] is not None):
            raise RuntimeError("Timer.start: %s timer is already ticking!" % name)
        self.starts[idx] = perf_counter()
        return
    
    def end(self, name):
        assert(name in self.names)
        
        idx = self.names[name]
        if (self.starts[idx] is None):
            raise RuntimeError("Timer.end: %s start time is not measured yet!" % name)

        self.times[idx] += perf_counter() - self.starts[idx]
        self.calls[idx] += 1
        self.starts[idx] = None
        return
    
    def print(self):
        print("Function name\tCalls\tTotal time\tTime/call\n")
        for name, idx in self.names.items():
            print("%s\t%d\t%.3e\t%.3e\n" % (name, self.calls[idx], self.times[idx], self.times[idx] / self.calls[idx]))
        return
    
    def export(self):
        for start in self.starts:
            if (start is not None):
                raise RuntimeError('Timer.export: cannot export while Timer is still ticking!')

        param_dict = {}
        param_dict["names"] = self.names
        param_dict["calls"] = self.calls
        param_dict["times"] = self.times
        return param_dict
    
    def load(self, dict_):
        self.names = dict_['names']
        self.calls = dict_['calls']
        self.times = dict_['times']

        assert(len(self.names) == len(self.calls))
        assert(len(self.names) == len(self.times))
        self.starts = [None] * len(self.names)
        return