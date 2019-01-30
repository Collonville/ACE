import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import win_unicode_console


Ml = np.matrix([[0.7126, 0.0236, 0.0217],
                [0.1142, 0.3976, 0.0453],
                [0.0827, 0.0256, 0.5512]], dtype=np.float)

Ml = np.power(Ml, 0.7)
Ml = Ml / np.max(Ml, axis=0)


print(Ml)
print(np.uint8(Ml * 255))

