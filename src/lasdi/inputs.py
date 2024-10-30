# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from warnings import warn

verbose : bool = False


# -------------------------------------------------------------------------------------------------
# Input Parser class
# -------------------------------------------------------------------------------------------------

class InputParser:
    """
    A InputParser objects acts as a wrapper around a dictionary of settings. Thus, each setting is 
    a key and the corresponding value is the setting's value. Because one setting may itself be 
    a dictionary (we often group settings; each group has a name but several constituent settings),
    the underlying dictionary is structured as a sequence of nested dictionaries. This class allows 
    the user to select a specific setting from that structure by specifying (via a list of strings) 
    where in that nested structure the desired setting lives. 
    """
    dict_   : dict  = None
    name    : str   = ""



    def __init__(self, dict_ : dict, name : str = "") -> None:
        """"
        Initializes an InputParser object by setting the underlying dictionary of settings as an 
        attribute.

        
        -------------------gi------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        dict_: The dictionary of settings. To avoid the risk of the user accidentally changing one 
        of the settings after wrapping it, we store a deep copy of dict in self.

        name: A string that we use toe name the InputParser object. We use this name when 
        reporting errors (it is purely for debugging purposes).
        """

        # A shallow copy could cause issues if the user changes dict's keys/values after 
        # initializing this object. We store a deep copy to avoid this risk.
        from copy import deepcopy
        self.dict_  : dict  = deepcopy(dict_)

        self.name   : str   = name



    def getInput(self, keys : list, fallback = None, datatype = None):
        '''
        A InputParser object acts as a wrapper around a dictionary of settings. That is, self.dict_
        is structured as a nested family of dictionaries. Each setting corresponds to a key in 
        self.dict_. The setting's value is the corresponding value in self.dict_. In many cases, 
        a particular setting may be nested within others. That is, a setting's value may itself be
        another dictionary housing various sub-settings. This function allows us to fetch a 
        specific setting from this nested structure. 

        Specifically, we specify a list of strings. keys[0] should be a key in self.dict_
        If so, we set val = self.dict_[keys[0]]. If there are more keys, then val should be a 
        dictionary and keys[1] should be a key in this dictionary. In this case, we replace val 
        with val[key[1]] and so on. This continues until we have exhausted all keys. There is one  
        important exception: 

            If at some point in the process, there are more keys but val is not a dictionary, or if
            there are more keys and val is a dictionary but the next key is not a key in that
            dictionary, then we return the fallback value. If the fallback value does not exist, 
            returns an error.
        

                       
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        keys: A list of keys we want to fetch from self.dict. keys[0] should be a key in self.dict_
        If so, we set val = self.dict_[keys[0]]. If there are more keys, then val should be a 
        dictionary and keys[1] should be a key in this dictionary. In this case, we replace val 
        with val[key[1]] and so on. This continues until we have exhausted all keys.

        fallback: A sort of default value. If at some point, val is not a dictionary (and there are
        more keys) or val is a dictionary but the next key is not a valid key in that dictionary, 
        then we return the fallback value.

        datatype: If not None, then we require that the final val has this datatype. If the final 
        val does not have the desired datatype, we raise an exception.

        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        The final val value as outlined by the process described above.
        '''

        # Concatenate the keys together. This is for debugging purposes.
        keyString = ""
        for key_ in keys:
            keyString += key_ + "/"

        # Begin by initializing val to self.dict_
        val = self.dict_

        # Cycle through the keys
        for key in keys:

            # Check if the current key is a key in val (this assumes val is a dictionary). If so, 
            # update val. Otherwise, return the fallback (if it is present) or raise an exception.
            if key in val:
                val = val[key]
            elif (fallback != None):
                return fallback
            else:
                raise RuntimeError("%s does not exist in the input dictionary %s!" % (keyString, self.name))
        
        # Check if the fallback and final val have the same type.
        if (fallback != None):
            if (type(val) != type(fallback)):
                warn("%s does not match the type with the fallback value %s!" % (str(type(val)), str(type(fallback))))
        
        # Check thast the final val matches the desired datatype
        if (datatype != None):
            if (type(val) != datatype):
                raise RuntimeError("%s does not match the specified datatype %s!" % (str(type(val)), str(datatype)))
        else:
            if verbose: warn("InputParser Warning: datatype is not checked.\n key: %s\n value type: %s" % (keys, type(val)))
        
        # All done!
        return val



# -------------------------------------------------------------------------------------------------
# Unused: getDictFromList function
# -------------------------------------------------------------------------------------------------

#def getDictFromList(list_, inputDict):
#    '''
#        get a dict with {key: val} from a list of dicts
#        NOTE: it returns only the first item in the list,
#            even if the list has more than one dict with {key: val}.
#    '''
#    dict_ = None
#    for item in list_:
#        isDict = True
#        for key, val in inputDict.items():
#            if key not in item:
#                isDict = False
#                break
#            if (item[key] != val):
#                isDict = False
#                break
#        if (isDict):
#            dict_ = item
#            break
#    if (dict_ == None):
#        raise RuntimeError('Given list does not have a dict with {%s: %s}!' % (key, val))
#    return dict_