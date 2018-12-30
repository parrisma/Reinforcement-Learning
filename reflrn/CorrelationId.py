from reflrn.Interface.UniversalCorrelationId import UniversalCorrelationId
from reflrn.UniqueId import UniqueId


#
# Allow for telemetry, files and events to be correlated together. The unique id is at class level so
# all correlation objects share same correlation id = session level correlation
#

class CorrelationId(UniversalCorrelationId):
    __unique_id = None

    def __init__(self):
        self.__unique_id = UniqueId.generate_id()

    def correlation_file_name(self,
                              file_root: str,
                              file_decorator: str,
                              file_extension: str,
                              file_sep: str = '_',
                              file_dir_sep='/') -> str:
        fr = ''
        if file_root is not None and len(file_root) > 0:
            fr = file_root + file_dir_sep + file_sep

        return fr + file_decorator + file_sep + self.__unique_id + file_sep + '.' + file_extension

    def id(self) -> str:
        return self.__unique_id
