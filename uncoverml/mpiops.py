import logging
import pickle

import numpy as np
from mpi4py import MPI

log = logging.getLogger(__name__)

# We're having trouble with the MPI pickling and 64bit integers
MPI.pickle.dumps = pickle.dumps
MPI.pickle.loads = pickle.loads

comm = MPI.COMM_WORLD
"""module-level MPI 'world' object representing all connected nodes
"""

chunks = comm.Get_size()
"""int: the total number of nodes in the MPI world
"""

chunk_index = comm.Get_rank()
"""int: the index (from zero) of this node in the MPI world. Also known as
the rank of the node.
"""


def run_once(f, *args, **kwargs):
    """Run a function on one node, broadcast result to all
    This function evaluates a function on a single node in the MPI world,
    then broadcasts the result of that function to every node in the world.
    Parameters
    ----------
    f : callable
        The function to be evaluated. Can take arbitrary arguments and return
        anything or nothing
    args : optional
        Other positional arguments to pass on to f
    kwargs : optional
        Other named arguments to pass on to f
    Returns
    -------
    result
        The value returned by f
    """
    if chunk_index == 0:
        f_result = f(*args, **kwargs)
    else:
        f_result = None
    result = comm.bcast(f_result, root=0)
    return result


def sum_axis_0(x, y, dtype):
    s = np.ma.sum(np.ma.vstack((x, y)), axis=0)
    return s


def max_axis_0(x, y, dtype):
    s = np.amax(np.array([x, y]), axis=0)
    return s


def min_axis_0(x, y, dtype):
    s = np.amin(np.array([x, y]), axis=0)
    return s


def unique(sets1, sets2, dtype):
    per_dim = zip(sets1, sets2)
    out_sets = [np.unique(np.concatenate(k, axis=0)) for k in per_dim]
    return out_sets

unique_op = MPI.Op.Create(unique, commute=True)
sum0_op = MPI.Op.Create(sum_axis_0, commute=True)
max0_op = MPI.Op.Create(max_axis_0, commute=True)
min0_op = MPI.Op.Create(min_axis_0, commute=True)


def count(x):
    x_n_local = np.ma.count(x, axis=0)
    x_n = comm.allreduce(x_n_local, op=sum0_op)
    still_masked = np.ma.count_masked(x_n)
    if still_masked != 0:
        raise ValueError("Can't compute count: subcounts are still masked")
    if hasattr(x_n, 'mask'):
        x_n = x_n.data
    return x_n


def mean(x):
    x_n = count(x)
    x_sum_local = np.ma.sum(x, axis=0)
    x_sum = comm.allreduce(x_sum_local, op=sum0_op)
    still_masked = np.ma.count_masked(x_sum)
    if still_masked != 0:
        raise ValueError("Can't compute mean: At least 1 column has no data")
    if hasattr(x_sum, 'mask'):
        x_sum = x_sum.data
    mean = x_sum / x_n
    return mean


def sd(x):
    x_mean = mean(x)
    delta_mean = mean((x - x_mean)**2)
    sd = np.sqrt(delta_mean)
    return sd


def eigen_decomposition(x):
    x_n = count(x)
    x_outer_local = np.ma.dot(x.T, x)
    outer = comm.allreduce(x_outer_local)
    cov = outer / x_n
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvals, eigvecs
