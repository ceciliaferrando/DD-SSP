import itertools
import numpy as np
from hdmm.workload import Identity, Prefix, Kronecker
from functools import reduce

class Workload(list):
    """
    A workload contains information about the queries/statistics that the synthetic data is expected to preserve.  It consists of a list of (proj, Q) pairs where
        - proj is a subset of attributes 
        - Q is a query matrix defined over the proj marginal
            (can be a numpy arrray, scipy sparse matrix, or linear operator)
    """
    def __init__(self, queries):
        self.queries = queries
        list.__init__(self, queries)

    def subset(self, num_queries, prng=np.random):
        if num_queries is None or num_queries >= len(self):
            return self
        idx = prng.choice(len(self.queries), size=num_queries, replace=False)
        new_queries = [self.queries[i] for i in idx]
        return Workload(new_queries)

    def cliques(self):
        return [M[0] for M in self.queries]

    def answer(self, data, breakdown=False):
        """
        Compute workload query answers on the dataset
        
        :param data: an mbi.Dataset object
                        can also pass in any object that supports "project" + "datavector"
        :param breakdown: flag to break down answers by subworkload
        """
        ans = []
        for proj, Q in self.queries:
            mu = data.project(proj).datavector()
            y = Q @ mu
            ans.append(y)
        if breakdown:
            print("ans", ans)
            return ans
        print("ans", ans)
        return np.concatenate(ans)

    def average_l1_error(self, true_data, synth_data):
        return self.error(true_data, synth_data) / len(self.queries)

    def compute_errors(self, true_data, synth_data):
        true_answers = self.answer(true_data)
        synth_answers = self.answer(synth_data)
    
        errors = {}
        errors['l1_error'] = np.linalg.norm(true_answers - synth_answers, 1) / true_data.df.shape[0] / len(self)
        errors['l2_error'] = np.linalg.norm(true_answers - synth_answers, 2) / true_data.df.shape[0] / len(self)
        errors['max_error'] = np.linalg.norm(true_answers - synth_answers, np.inf) / true_data.df.shape[0]
        return errors

    def error(self, true_data, synth_data, ord=1, breakdown=False, normalize=True):
        """
        Compute the Lp error between true workload answers and synthetic workload answers
    
        :param true_data: an mbi.Dataset object
        :param synth_data: an mbi.Dataset object
                            can also pass in any object that supports "project" + "datavector"
        :param ord: the order of the norm
        :param breakdown: flag to break down error by subworkload
        :param normalize: normalize error by number of records
        """
        ans1 = self.answer(true_data, breakdown)
        ans2 = self.answer(synth_data, breakdown)
        if breakdown:
            ans = np.array([np.linalg.norm(a1-a2, ord) for a1, a2 in zip(ans1, ans2)])
        else: 
            ans = np.linalg.norm(ans1 - ans2, ord)
        if normalize:
            ans /= true_data.records
        return ans

    def __add__(self, other):
        return Workload(self.queries + other.queries)

    def __len__(self):
        return len(self.queries)

def MarginalQuery(domain):
    return Kronecker([Identity(n) for n in domain.shape])

def PrefixMarginalQuery(domain):
    def sub(n):
        return Prefix(n) if n in [100, 101] else Identity(n)
    return Kronecker([sub(n) for n in domain.shape])

def from_cliques(domain, cliques, Query=MarginalQuery):
    queries = []
    for proj in cliques:
        Q = Query(domain.project(proj))
        queries.append((proj, Q))
    return Workload(queries)

def all_kway(domain, k, threshold=np.inf, Query=MarginalQuery):
    """
    Produces a marginal query workload, with option to have non-identity queries on each marginal
        
    :param domain: the dataset domain
    :parma k: the number of attributes in the marginal 
                (can be an int or a list of int)
    :param threshold: threshold for filtering large marginals.  Marginals defined over domain 
                        larger than threshold will be filtered out
    :param Query: a function mapping domain for a marginal to a query set 
                    (e.g., MarginalQuery or PrefixMarginalQuery)
    """
    if type(k) is list:
        return reduce(Workload.__add__, [all_kway(domain,l,threshold,Query) for l in  k])
    queries = []
    for proj in itertools.combinations(domain, k):
        if domain.size(proj) <= threshold:
            Q = Query(domain.project(proj))
            queries.append((proj, Q))
    return Workload(queries)

def target_kway(domain, k, targets, threshold=np.inf, Query=MarginalQuery):
    """
    Produces a marginal query workload, with option to have non-identity queries on each marginal
        
    :param domain: the dataset domain
    :parma k: the number of attributes in the marginal 
                (can be an int or a list of int)
    :param threshold: threshold for filtering large marginals.  Marginals defined over domain 
                        larger than threshold will be filtered out
    :param Query: a function mapping domain for a marginal to a query set 
                    (e.g., MarginalQuery or PrefixMarginalQuery)
    """
    if type(targets) is str:
        targets = (targets,)
    if type(k) is list:
        return reduce(Workload.__add__, [target_kway(domain,l,targets,threshold,Query) for l in  k])
    queries = []
    for proj in itertools.combinations(domain.invert(targets), k):
        cl = proj + targets
        if domain.size(cl) <= threshold:
            Q = Query(domain.project(cl))
            queries.append((cl, Q))
    return Workload(queries)

def weighted_kway(domain,k,number,threshold=np.inf,Query=MarginalQuery,prng=np.random.RandomState(0)):
    """
    Produces a marginal query workload, with a given number of marginals chosen randomly. 
        
    :param domain: the dataset domain
    :parma k: the number of attributes in the marginal 
                (can be an int or a list of int)
    :param number: the number of marginals to include in the workload
    :param threshold: threshold for filtering large marginals.  Marginals defined over domain 
                        larger than threshold will be filtered out
    :param Query: a function mapping domain for a marginal to a query set 
                    (e.g., MarginalQuery or PrefixMarginalQuery)
    :param prng: a pseudo-random number generator
    """
    probas = prng.exponential(scale=1, size=len(domain))**2
    probas /= probas.sum()

    workload = []
    while len(workload) < number:
        keys = sorted(prng.choice(len(domain), size=k, replace=False, p=probas))
        cl = tuple(domain.attrs[i] for i in keys)
        if domain.size(cl) <= threshold:
            workload.append(cl)
    
    return from_cliques(domain, workload, Query)

def random_kway(domain,k,number,threshold=np.inf,Query=MarginalQuery,prng=np.random.RandomState(0),weighted=False):
    """
    Produces a marginal query workload, with a given number of marginals chosen randomly. 
        
    :param domain: the dataset domain
    :parma k: the number of attributes in the marginal 
                (can be an int or a list of int)
    :param number: the number of marginals to include in the workload
    :param threshold: threshold for filtering large marginals.  Marginals defined over domain 
                        larger than threshold will be filtered out
    :param Query: a function mapping domain for a marginal to a query set 
                    (e.g., MarginalQuery or PrefixMarginalQuery)
    :param prng: a pseudo-random number generator
    :param weighted: flag to also assign random weights to each marginal query
    """
    W = all_kway(domain, k, threshold, Query)
    assert number <= len(W.queries)
    idx = prng.choice(len(W.queries), number, replace=False)
    queries = []
    for i in idx:
        proj, Q = W[i]
        if weighted:
            w = 2.0*prng.rand()
            Q = w * Q
        queries.append((proj, Q))
    return Workload(queries)
    
