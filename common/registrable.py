import logging

from collections import defaultdict

logger = logging.getLogger(__name__)


class Registrable:
    _registry = defaultdict(dict)

    @classmethod
    def register(cls, name):
        registry = Registrable._registry[cls]

        logger.debug("Register " + name)

        def add_subclass_to_registry(subclass):
            logger.debug("Adding" + str(subclass) + "to" + str(cls) + "registry with name " + name)
            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls, name):
        logger.debug("By name" + str(cls) + " name=" + name)
        if name not in Registrable._registry[cls]:
            raise ValueError
        return Registrable._registry[cls].get(name)
