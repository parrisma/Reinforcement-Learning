from reflrn.Interface.ModelParams import ModelParams


#
# Mediator for all common NN model params
#

class GeneralModelParams(ModelParams):

    def __init__(self,
                 params: [str, object]):
        self.__plist = dict()
        for p in params:
            self.__plist[p[0]] = p[1]
        return

    #
    # Return the parameter(s) requested with the given parameter key(s)
    #
    # If just one parameter is requested then the param value is returned
    # else a list of values is returned.
    #
    def get_parameter(self,
                      required_params: [str]):
        pout = None
        if isinstance(required_params, str):
            required_params = [required_params]
        for rp in required_params:
            if rp not in self.__plist:
                raise ModelParams.RequestedParameterNotAvailable('No such parameter: ' + rp)
            if pout is None:
                pout = list()
            pout.append(self.__plist[rp])

        if len(pout) > 1:
            return pout
        else:
            return pout[0]

    #
    # Override or Augment the current parameter set with the given parameters.
    #
    def override_parameters(self, params: [[str, object]]) -> None:
        if isinstance(params, list):
            if isinstance(params[0], list) or isinstance(params[0], tuple):
                for p in params:
                    self.__plist[p[0]] = p[1]
            else:
                raise ValueError("Invalid type of over-ride parameters passed expected list of tuple or tuple")
        else:
            if isinstance(params[0], list) or isinstance(params[0], tuple):
                self.__plist[params[0]] = params[1]
            else:
                raise ValueError("Invalid type of over-ride parameters passed expected list of tuple or tuple")
        return

    #
    # Over-ride iter ..so we can be cast to list etc.
    #
    def __iter__(self):
        kys = self.__plist
        for k in kys:
            yield k, self.__plist[k]
