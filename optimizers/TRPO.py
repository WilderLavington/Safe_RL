
import torch
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import copy

class TRPO(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, divergence_limit=10,
                 starting_step_size = 1):
        # set the starting step size
        self.current_step_size = starting_step_size
        # set the lagrange multiplier for divergence
        self.divergence_limit = divergence_limit
        # set the epsilon for improvement
        self.epsilon = 0.01
        # the rest is from SGD
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(TRPO, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TRPO, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, H, step_size=1, closure=None):

        # literally no idea what this does
        loss = None
        if closure is not None:
            loss = closure()

        # set parameters
        params = [p for p in self.param_groups[0]['params']]
        grads = [p.grad for p in params]

        # convert parameters to a vector
        param_vector = parameters_to_vector(params)
        grad_vector = parameters_to_vector(grads)

        # apply rotation / contract / expansion
        soln,_ = torch.solve(grad_vector.unsqueeze(1).unsqueeze(0), H.unsqueeze(0))
        scaled_gradient = soln[0].reshape(-1)

        # add the charactoristic scaling
        scaling = torch.dot(scaled_gradient,soln.reshape(-1))
        scaled_gradient *= step_size*torch.sqrt(self.divergence_limit/(scaling+self.epsilon))

        # check that the scaling is ok before updating parameters
        if scaling > 0.:
            # update the gradient weights
            vector_to_parameters(scaled_gradient, grads)

        # now we can perform the update
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
