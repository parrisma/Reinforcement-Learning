import uuid

from reflrn.Interface.UniversallyUniqueId import UniversallyUniqueId


#
# Generate a universally unique (almost) id that can be used in file names (temp) etc
#

class UniqueId(UniversallyUniqueId):
    @classmethod
    def generate_id(cls) -> str:
        return str(uuid.uuid4())
