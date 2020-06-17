from jax import random
from jax import lax
import jax.numpy as np
import jax.ops as ops

key = random.PRNGKey(0)

# Conv layer shape
N = 2 # batch
C = 3 # input channels
H = 6 # height
W = 8 # width

# Convolution filter shape
K = 5 # output channels
R = 3 # vertical kernel_size
S = 3 # horizontal kernel_size

# Weight "algebra" elements have shape [ac].
# Activation "algebra" elements have shape [cb].
# Shape must match the mat_*_rule, below.
a = 2
c = 2
b = 2

# Initialize filter-weights, inputs, and outputs.
F = random.normal(key, (K,C,R,S,a*c))
x = random.normal(key, (N,C,H,W,c*b))
y = np.zeros((N,K,H,W,a*b))

# Let's compute the conv2d layer output using the formulation in the
# "AlgebraNets" paper Appendix E: Convolution.

# Indices map to matrix elements like:
#   0 1
#   2 3
mat_22_rule = [[(0, 0), (1, 2)],
               [(0, 1), (1, 3)],
               [(2, 0), (3, 2)],
               [(2, 1), (3, 3)]]
for i in range(a*c):
    for j in range(b):
        xij = x[:,:,:,:, mat_22_rule[i][j][1]]
        Wij = F[:,:,:,:, mat_22_rule[i][j][0]]
        yij = lax.conv(xij, Wij, (1,1), 'SAME') + y[:,:,:,:,i]
        y = ops.index_update(y, ops.index[:,:,:,:,i], yij)

# Let's compute it again using a single conv2d operator:

F = np.reshape(F, (K,C,R,S,a,c))
F = np.einsum('KCRSac->KaCcRS', F)
F = np.reshape(F, (K*a, C*c, R, S))

x = np.reshape(x, (N,C,H,W,c,b))
x = np.einsum('NCHWcb->NbCcHW', x)
x = np.reshape(x, (N*b, C*c, H, W))

yy = lax.conv(x, F, (1,1), 'SAME')

yy = np.reshape(yy, (N, b, K, a, H, W))
yy = np.einsum('NbKaHW->NKHWab', yy)
yy = np.reshape(yy, (N,K,H,W,a*b))

# Compare results:
diff = yy-y
abs_diff = np.abs(diff)
max_diff = np.max(abs_diff)
print("max_error: {:f}".format(max_diff))
