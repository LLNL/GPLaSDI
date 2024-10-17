from warnings import warn

verbose = False

class InputParser:
    dict_ = None
    name = ""

    def __init__(self, dict, name = ""):
        from copy import deepcopy
        self.dict_ = deepcopy(dict)
        self.name = name
        return

    def getInput(self, keys, fallback=None, datatype=None):
        '''
            Find the value corresponding to the list of keys.
            If the specified keys do not exist, use the fallback value.
            If the fallback value does not exist, returns an error.
            If the datatype is specified, enforce the output value has the right datatype.
        '''
        keyString = ""
        for key_ in keys:
            keyString += key_ + "/"

        val = self.dict_
        for key in keys:
            if key in val:
                val = val[key]
            elif (fallback != None):
                return fallback
            else:
                raise RuntimeError("%s does not exist in the input dictionary %s!" % (keyString, self.name))

        if (fallback != None):
            if (type(val) != type(fallback)):
                warn("%s does not match the type with the fallback value %s!" % (str(type(val)), str(type(fallback))))
                
        if (datatype != None):
            if (type(val) != datatype):
                raise RuntimeError("%s does not match the specified datatype %s!" % (str(type(val)), str(datatype)))
        else:
            if verbose: warn("InputParser Warning: datatype is not checked.\n key: %s\n value type: %s" % (keys, type(val)))
        return val

def getDictFromList(list_, inputDict):
    '''
        get a dict with {key: val} from a list of dicts
        NOTE: it returns only the first item in the list,
        even if the list has more than one dict with {key: val}.
    '''
    dict_ = None
    for item in list_:
        isDict = True
        for key, val in inputDict.items():
            if key not in item:
                isDict = False
                break
            if (item[key] != val):
                isDict = False
                break
        if (isDict):
            dict_ = item
            break
    if (dict_ == None):
        raise RuntimeError('Given list does not have a dict with {%s: %s}!' % (key, val))
    return dict_
