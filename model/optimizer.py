import torch
from lib.tensorlist import TensorList


class MinimizationProblem:
    def __call__(self, x: TensorList) -> TensorList:
        """Shall compute the residuals."""
        raise NotImplementedError

    def ip_input(self, a, b):
        """Inner product of the input space."""
        return sum(a.view(-1) @ b.view(-1))

    def M1(self, x):
        return x


class GaussNewtonCG:

    def __init__(self, problem: MinimizationProblem, variable: TensorList, cg_eps=0.0, fletcher_reeves=True,
                 standard_alpha=True, direction_forget_factor=0, step_alpha=1.0):

        self.fletcher_reeves = fletcher_reeves
        self.standard_alpha = standard_alpha
        self.direction_forget_factor = direction_forget_factor

        # State
        self.p = None
        self.rho = torch.ones(1)
        self.r_prev = None

        # Right hand side
        self.b = None

        self.problem = problem
        self.x = variable

        self.cg_eps = cg_eps
        self.f0 = None
        self.g = None
        self.dfdxt_g = None

        self.residuals = torch.zeros(0)
        self.external_losses = []
        self.internal_losses = []
        self.gradient_mags = torch.zeros(0)

        self.step_alpha = step_alpha

    def clear_temp(self):
        self.f0 = None
        self.g = None
        self.dfdxt_g = None

    def run(self, num_cg_iter, num_gn_iter=None):

        self.problem.initialize()

        if isinstance(num_cg_iter, int):
            if num_gn_iter is None:
                raise ValueError('Must specify number of GN iter if CG iter is constant')
            num_cg_iter = [num_cg_iter] * num_gn_iter

        num_gn_iter = len(num_cg_iter)
        if num_gn_iter == 0:
            return

        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for cg_iter in num_cg_iter:
            self.run_GN_iter(cg_iter)

        self.x.detach_()
        self.clear_temp()

        return self.external_losses, self.internal_losses, self.residuals

    def run_GN_iter(self, num_cg_iter):

        self.x.requires_grad_(True)

        self.f0 = self.problem(self.x)
        self.g = self.f0.detach()
        self.g.requires_grad_(True)
        self.dfdxt_g = TensorList(torch.autograd.grad(self.f0, self.x, self.g, create_graph=True))  # df/dx^t @ f0
        self.b = - self.dfdxt_g.detach()

        delta_x, res = self.run_CG(num_cg_iter, eps=self.cg_eps)

        self.x.detach_()
        self.x += self.step_alpha * delta_x
        self.step_alpha = min(self.step_alpha * 1.2, 1.0)

    def reset_state(self):
        self.p = None
        self.rho = torch.ones(1)
        self.r_prev = None

    def run_CG(self, num_iter, x=None, eps=0.0):
        """Main conjugate gradient method"""

        # Apply forgetting factor
        if self.direction_forget_factor == 0:
            self.reset_state()
        elif self.p is not None:
            self.rho /= self.direction_forget_factor

        if x is None:
            r = self.b.clone()
        else:
            r = self.b - self.A(x)

        # Loop over iterations
        for ii in range(num_iter):

            z = self.problem.M1(r)  # Preconditioner

            rho1 = self.rho
            self.rho = self.ip(r, z)

            if self.p is None:
                self.p = z.clone()
            else:
                if self.fletcher_reeves:
                    beta = self.rho / rho1
                else:
                    rho2 = self.ip(self.r_prev, z)
                    beta = (self.rho - rho2) / rho1

                beta = beta.clamp(0)
                self.p = z + self.p * beta

            q = self.A(self.p)
            pq = self.ip(self.p, q)

            if self.standard_alpha:
                alpha = self.rho / pq
            else:
                alpha = self.ip(self.p, r) / pq

            # Save old r for PR formula
            if not self.fletcher_reeves:
                self.r_prev = r.clone()

            # Form new iterate
            if x is None:
                x = self.p * alpha
            else:
                x += self.p * alpha

            if ii < num_iter - 1:
                r -= q * alpha

        return x, []

    def A(self, x):
        dfdx_x = torch.autograd.grad(self.dfdxt_g, self.g, x, retain_graph=True)
        return TensorList(torch.autograd.grad(self.f0, self.x, dfdx_x, retain_graph=True))

    def ip(self, a, b):
        return self.problem.ip_input(a, b)
