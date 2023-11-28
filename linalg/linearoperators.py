import scipy.sparse.linalg as sla
import pylops


class LowRankOperator(sla.LinearOperator):
    dtype = float

    def __init__(self, vector, value):
        self.vector = vector
        self.value = value
        self.shape = (vector.shape[0], vector.shape[0])

    def _matvec(self, x):
        return self.value * self.vector.dot(x) * self.vector

    def _rmatvec(self, x):
        return self._matvec(x)


def construct_linear_op_lowrank(vectors, values):
    list_op = []
    for i in range(vectors.shape[1]):
        _op = LowRankOperator(vectors[:, i], values[i])
        if i == 0:
            op = _op
        else:
            op += _op
        list_op.append(_op)
    return pylops.aslinearoperator(op)  #


class SumLowRankOp(sla.LinearOperator):
    dtype = float

    def __init__(self, vectors, values):
        self.shape = (vectors.shape[0], vectors.shape[0])
        self.vectors = vectors
        self.values = values
        self.rank = vectors.shape[1]
        self.list_lrop = []
        for i in range(self.rank):
            self.list_lrop.append(LowRankOperator(self.vectors[:, i], self.values[i]))

    def _matvec(self, x):
        return super()._matvec(x)
