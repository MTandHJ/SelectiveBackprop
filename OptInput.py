'''
OptInput.py can be replaced by argparse module.
'''




class Unit:

    def __init__(self, command, type=str,
                    default=None):
        if default is None:
            default = type()
        self.command = command
        self.type = type
        self.default = default

class Opi:
    """
    >>> parser = Opi()
    >>> parser.add_opt(command="lr", type=float)
    >>> parser.add_opt(command="epochs", type=int)
    """
    def __init__(self):
        self.store = []
        self.infos = {}

    def add_opt(self, **kwargs):
        self.store.append(
            Unit(**kwargs)
        )

    def acquire(self):
        s = "Acquire args {0.command} [" \
            "type:{0.type.__name__} " \
            "default:{0.default}] : "
        for unit in self.store:
            while True:
                inp = input(s.format(
                    unit
                ))
                try:
                    if inp: #若有输入
                        inp = unit.type(inp)
                    else:
                        inp = unit.default
                    self.infos.update(
                        {unit.command:inp}
                    )
                    self.__setattr__(unit.command, inp)
                    break
                except:
                    print("Type {0} should be given".format(
                        unit.type.__name__
                    ))


if __name__ == "__main__":
    parser = Opi()
    parser.add_opt(command = "x", type=int)
    parser.add_opt(command="y", type=str)
    parser.acquire()
    print(parser.infos)
    print(parser.x)
