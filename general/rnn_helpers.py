
import numpy as np
import itertools

def multidim_gaussian(x, cov):
    num = np.dot(np.dot(x, np.linalg.inv(cov)), x)
    return np.exp(-num)

def gaussian_basefunc(dims, usedims=None, periodicity=None):
    if periodicity is None:
        periodicity = np.zeros(dims)
    if usedims is None:
        usedims = np.ones(dims, dtype=bool)
    periodicity = np.array(periodicity)
    usedims = np.array(usedims)
    def gbf(x, cov):
        x = np.array(x)
        cov = np.array(cov)
        x_p = x + periodicity/2
        pm = periodicity > 0
        x_p[pm] = x_p[pm] % periodicity[pm]
        x_p = x_p - periodicity / 2
        if cov.size == dims:
            fcov = np.zeros((dims, dims))
            fcov[np.diag_indices(dims, ndim=2)] = cov
        else:
            fcov = cov
        x_p = x_p[usedims]
        fcov1 = fcov[usedims, :]
        fcov2 = fcov1[:, usedims]
        resp = multidim_gaussian(x_p, fcov2)
        return resp
    return gbf

class Receptor(object):
    
    def __init__(self, base_func, loc, wid):
        self.func = lambda x: base_func(np.array(x) - loc, wid)
        self.bf = base_func
        self.loc = loc
        self.wid = wid
        
    def sample(self, x):
        return self.func(x)

class MultiLattice(object):
    
    def __init__(self, list_of_latts):
        self.latts = list_of_latts
        self.n_latts = len(list_of_latts)
        self.n_receptors = np.sum([latt.n_receptors for latt in list_of_latts])

    def sample(self, x):
        assert len(x) == self.n_latts
        resp = np.zeros(self.n_receptors)
        last_end = 0
        for i, latt in enumerate(self.latts):
            resp[last_end:last_end + latt.n_receptors] = latt.sample(x[i])
            last_end = last_end + latt.n_receptors
        return resp

class ReceptorLattice(object):

    def __init__(self, n, delta, wid, space_sizes, base_func, loc_jitter=None, 
                 wid_jitter=None):
        """
        Initialize a receptor array, which produces an array of responses (from
        different receptors) to a given stimulus. 
        
        Parameters
        ----------
        n : int
            Number of duplicates for each tuning of receptor
        delta : tuple of floats
            Average spacing between receptors (len M, where M is the 
            dimensionality of the space)
        wid : tuple of floats or array
            Average width of receptors (len M); or array of shape (M, M),
            the covariance matrix
        space_sizes : tuple of floats
            Size of dimensions (len M)
        base_func : function
            M-dimensional function, with a parameter for width, signature 
            f(x, w) --> i, where x is a len M tuple and w is a scalar parameter, i
            is the intensity value of the response to a stimulus at location x. The 
            base_func should, itself, contain any desired periodicity on any of the
            dimensions.
        loc_jitter : func or None
            A function that returns some M-dimensional vector that is added to
            the location of each receptor, will be called once for each receptor.
            The values in the vector will be divided by the space_sizes.
        wid_jitter : func or None
            Same as loc_jitter, but applied to the width of the receptors.

        """
        self.delta = np.array(delta)
        self.space_sizes = np.array(space_sizes)
        self.n_receptors_dim = np.floor(self.space_sizes / self.delta)
        ind_ranges = [range(0, int(x)) for x in self.n_receptors_dim]
        receptor_inds = itertools.product(*ind_ranges)
        self.n_receptors = int(np.array(np.product(self.n_receptors_dim))*n)
        self.receptors = []
        self.locs = []
        self.wids = []
        for i, ri in enumerate(receptor_inds):
            for j in range(n):
                loc = np.array(ri)*self.delta + self.delta/2
                if loc_jitter is not None:
                    loc = loc + loc_jitter()/self.space_sizes
                if wid_jitter is not None:
                    r_wid = wid + wid_jitter()/self.space_sizes
                else:
                    r_wid = wid
                receptor = Receptor(base_func, loc, r_wid)
                self.receptors.append(receptor)
                self.locs.append(loc)
                self.wids.append(r_wid)
        
    def sample(self, x):
        resp = np.zeros(self.n_receptors)
        if x is not None:
            for i, rec in enumerate(self.receptors):
                resp[i] = rec.sample(x)
        return resp

    def add_lattice(self, lattice):
        assert np.all(self.delta == lattice.delta)
        assert np.all(self.space_sizes == lattice.space_sizes)
        self.n_receptors_dim = self.n_receptors_dim + lattice.n_receptors_dim
        self.n_receptors = self.n_receptors + lattice.n_receptors
        self.receptors = self.receptors + lattice.receptors
        self.locs = self.locs + lattice.locs
        self.wids = self.wids + lattice.wids
