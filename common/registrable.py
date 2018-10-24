import logging

from collections import defaultdict

logger = logging.getLogger(__name__)


class Registrable:
    """
    Allows subclasses to register themselves into superclass registry.
    It's useful when we read class name from config (e.g. dataset name)
    and would like to create specific dataset class instance.
    With ``Registrable`` we can do this as follows:

    ``@BaseClass.by_name(config['name'])``

    To register, just decorate class definition with the classmethod
    ``@BaseClass.register(name)``.
    """
    _registry = defaultdict(dict)

    @classmethod
    def register(cls, name):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass):
            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls, name):
        logger.debug("By name" + str(cls) + " name=" + name)
        if name not in Registrable._registry[cls]:
            raise ValueError
        return Registrable._registry[cls].get(name)
