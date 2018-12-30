import abc


#
# Generation and management of correlation id's
#


class UniversalCorrelationId(metaclass=abc.ABCMeta):
    TELEMETRY_FILE_EXT = '.tml'
    KERAS_MODEL_FILE_EXT = '.pb'
    LOG_FILE_EXT = '.log'

    #
    # The universally unique correlation id of the correlation id (object)
    # - this is immutable and generated during construction.
    #
    @abc.abstractmethod
    def id(self) -> str:
        pass

    #
    # Correlation File Name - create a correlated file name
    #
    @abc.abstractmethod
    def correlation_file_name(self,
                              file_root: str,
                              file_decorator: str,
                              file_extension: str,
                              file_sep: str = '_',
                              file_dir_sep='/') -> str:
        pass
