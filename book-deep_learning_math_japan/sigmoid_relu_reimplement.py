# ANCHOR test anchor
# TODO study how to use new extension, refer to: https://github.com/StarlaneStudios/vscode-comment-anchors


import numpy as np


def sig(x):
    return 1 / (1+np.exp(-x))


x = 1.0
print('Applying Sigmoid Activation on (%.1f), gives %.1f' % (x, sig(x)))

x =-10.0
print('Applying Sigmoid Activation on (%.1f), gives %.1f' % (x, sig(x)))

x=0.0
print('Applying Sigmoid Activation on (%.1f), gives %.1f' % (x, sig(x)))