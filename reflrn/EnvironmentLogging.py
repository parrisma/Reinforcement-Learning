import logging
import sys


#
# Create simple console & file logger.
#


class EnvironmentLogging:
    __lg = None

    def __init__(self, env_name: str = 'root', log_file: str = '', level=logging.DEBUG):
        if len(log_file) == 0:
            log_file = "./" + self.__class__.__name__ + ".log"

        logging.basicConfig(level=level,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=log_file,
                            filemode='w')

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        self.__lg = logging.getLogger(env_name)
        self.__lg.addHandler(console)

    def get_logger(self):
        return self.__lg
