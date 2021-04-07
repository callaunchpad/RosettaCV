import torch
import torch.nn as nn
from torch.autograd import Variable

class ReptileModel(nn.Module):
	def __init__():
		super.__init__(self)
	
	# Manually change the gradients for the network, as expected in the
	# reptile algorithm, to the difference between the weights in self and net.
	def set_grads(self, net):
		for param, target_param in zip(self.parameters(), target.parameters()):
			if param.grad() is None:
				if self.is_cuda():
					param.grad = Variable(torch.zeros(param.size())).cuda()
				else:
					param.grad = Variable(torch.zeros(param.size))
			param.grad.data.zero_() # Zeroing out gradients again, to be sure.
			param.grad.data.add_(param.data - target_param.data)
	
	def is_cuda(self):
		return next(self.parameters()).is_cuda
