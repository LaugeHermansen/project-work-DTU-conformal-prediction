
import numpy as np
<<<<<<< HEAD



from collections import namedtuple

=======
a = np.array([[0,1],[1,0],[1,0],[1,1]]).astype(bool)
y = np.array([0,0,1,1])
print(a[np.arange(len(y)), y])
>>>>>>> re-structuring_CP_classes

Out = namedtuple('out', ["a","b","c"])
output = Out(2,7,5)

print(output)

