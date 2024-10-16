from enum import Enum

class NextStep(Enum):
    Initial         = 1
    Train           = 2
    RunSample       = 3
    CollectSample   = 4

class Result(Enum):
    Unexecuted      = 1
    Success         = 2
    Fail            = 3
    Complete        = 4