import abc


#
# Generate a universally Unique Id
#


class UniversallyUniqueId(metaclass=abc.ABCMeta):

    #
    # Generate a new universally unique id
    #
    @abc.abstractmethod
    def generate_id(self) -> str:
        pass
