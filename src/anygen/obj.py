class Obj:
    def __init__(self, *args, **kw):
        for d in args:
            self.__dict__.update(d)
        self.__dict__.update(kw)
