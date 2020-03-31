from torch import Tensor

def simplex(t: Tensor, axis=1) -> bool:
    """
    check if the matrix is the probability distribution
    :param t:
    :param axis:
    :return:
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, rtol=1e-4, atol=1e-4)

class Entropy(nn.Module):
    def __init__(self, reduce=True, eps=1e-16):
        super().__init__()
        r"""
        the definition of Entropy is - \sum p(xi) log (p(xi))
        """
        self.eps = eps
        self.reduce = reduce

    def forward(self, input: torch.Tensor):
        assert input.shape.__len__() >= 2
        b, _, *s = input.shape        input = F.softmax(input, dim=1)

        assert simplex(input)
        e = input * (input + self.eps).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, *s])
        if self.reduce:
            return e.mean()
        return e

# Simplifiy -
class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(1)
        return b.mean()


### Usage 
# from entropy_loss_pytorch import Entropy
# from entropy_loss_pytorch import EntropyLoss
