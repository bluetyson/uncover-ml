from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, \
    RationalQuadratic, Matern, WhiteKernel

kernel = C(0.1, (1e-4, 1e2)) * RBF(1, (1e-2, 1e2)) + \
   C(0.1, (1e-4, 1e2)) * RationalQuadratic(length_scale=0.01, alpha=0.1) + \
   C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(0.01)

