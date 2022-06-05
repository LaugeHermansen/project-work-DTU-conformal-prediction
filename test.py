import types
class hej:
    def __init__(self, a):
        self.a = a

h = hej(4)

vars()[type(h).__name__].__call__ = lambda self: print(self.a)
# hej.__call__ = lambda self: print(self.a)






h()
