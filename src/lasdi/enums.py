from enum import Enum

class NextStep(Enum):
    Train           = 1
    PickSample      = 2
    RunSample       = 3
    CollectSample   = 4

class Result(Enum):
    Unexecuted      = 1
    Success         = 2
    Fail            = 3
    Complete        = 4