import torch
from .tensorlist import TensorList


class MinimizationProblem:
    def __call__(self, x: TensorList) -> TensorList:
        """Shall compute the residuals."""
        raise NotImplementedError

    def ip_input(self, a, b):
        """Inner product of the input space."""
        return sum(a.view(-1) @ b.view(-1))

    def M1(self, x):
        return x

    def M2(self, x):
        return x


class ConjugateGradientBase:

    def __init__(self, fletcher_reeves=True, standard_alpha=True, direction_forget_factor=0, debug=False):
        self.fletcher_reeves = fletcher_reeves
        self.standard_alpha = standard_alpha
        self.direction_forget_factor = direction_forget_factor
        self.debug = debug

        # State
        self.p = None
        self.rho = torch.ones(1)
        self.r_prev = None

        # Right hand side
        self.b = None

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

        # Norms of residuals etc for debugging
        resvec = None
        if self.debug:
            normr = self.residual_norm(r)
            resvec = torch.zeros(num_iter + 1)
            resvec[0] = normr

        # Loop over iterations
        for ii in range(num_iter):

            z = self.M2(self.M1(r))

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

            if ii < num_iter - 1 or self.debug:
                r -= q * alpha

            if eps > 0.0 or self.debug:
                normr = self.residual_norm(r)

            if self.debug:
                self.evaluate_CG_iteration(x)
                resvec[ii + 1] = normr

            if eps > 0 and normr <= eps:
                if self.debug:
                    print('Stopped CG since norm smaller than eps')
                break

        if resvec is not None:
            resvec = resvec[:ii + 2]

        return x, resvec

    def A(self, x):
        # Left hand operation
        raise NotImplementedError

    def ip(self, a, b):
        # Implements the inner product
        return a.view(-1) @ b.view(-1)

    def residual_norm(self, r):
        res = self.ip(r, r).sum()
        if isinstance(res, (TensorList, list, tuple)):
            res = sum(res)
        return res.sqrt()

    def check_zero(self, s, eps=0.0):
        ss = s.abs() <= eps
        if isinstance(ss, (TensorList, list, tuple)):
            ss = sum(ss)
        return ss.item() > 0

    def M1(self, x):
        return x

    def M2(self, x):
        return x

    def evaluate_CG_iteration(self, x):
        pass


class GaussNewtonCG(ConjugateGradientBase):

    def __init__(self, problem: MinimizationProblem, variable: TensorList, cg_eps=0.0, fletcher_reeves=True,
                 standard_alpha=True, direction_forget_factor=0, debug=False, analyze=False, step_alpha=1.0):

        super().__init__(fletcher_reeves, standard_alpha, direction_forget_factor, debug or analyze)

        self.problem = problem
        self.x = variable

        self.analyze_convergence = analyze

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

    def run_GN(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, num_cg_iter, num_gn_iter=None):

        if isinstance(num_cg_iter, int):
            if num_gn_iter is None:
                raise ValueError('Must specify number of GN iter if CG iter is constant')
            num_cg_iter = [num_cg_iter] * num_gn_iter

        num_gn_iter = len(num_cg_iter)
        if num_gn_iter == 0:
            return

        if self.analyze_convergence:
            self.evaluate_CG_iteration(0)

        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for cg_iter in num_cg_iter:
            self.run_GN_iter(cg_iter)

        # print("self.problem.num_ips", self.problem.num_ips)
        if self.debug:
            if not self.analyze_convergence:
                self.f0 = self.problem(self.x)
                f0 = self.f0.detach()
                external_loss, internal_loss = self.problem.ip_debug(f0, f0)
                # loss = self.problem.ip_input(self.g, self.g)
                self.external_losses.append(external_loss)
                self.internal_losses.append(internal_loss)

        self.x.detach_()
        self.clear_temp()

        return self.external_losses, self.internal_losses, self.residuals

    def run_GN_iter(self, num_cg_iter):

        self.x.requires_grad_(True)

        self.f0 = self.problem(self.x)
        self.g = self.f0.detach()

        if self.debug and not self.analyze_convergence:
            external_loss, internal_loss = self.problem.ip_debug(self.g, self.g)
            self.external_losses.append(external_loss)
            self.internal_losses.append(internal_loss)

        self.g.requires_grad_(True)

        # Get df/dx^t @ f0
        self.dfdxt_g = TensorList(torch.autograd.grad(self.f0, self.x, self.g, create_graph=True))

        # Get the right hand side
        self.b = - self.dfdxt_g.detach()

        # Run CG
        delta_x, res = self.run_CG(num_cg_iter, eps=self.cg_eps)

        self.x.detach_()
        self.x += self.step_alpha * delta_x
        self.step_alpha = min(self.step_alpha * 1.2, 1.0)

        if self.debug:
            self.residuals = torch.cat((self.residuals, res))

    def A(self, x):
        # out2 = TensorList(torch.autograd.grad(self.a, self.x2, x, retain_graph=True))
        dfdx_x = torch.autograd.grad(self.dfdxt_g, self.g, x, retain_graph=True)
        return TensorList(torch.autograd.grad(self.f0, self.x, dfdx_x, retain_graph=True))

    def ip(self, a, b):
        # Implements the inner product
        return self.problem.ip_input(a, b)

    def M1(self, x):
        return self.problem.M1(x)

    def M2(self, x):
        return self.problem.M2(x)

    def evaluate_CG_iteration(self, delta_x):
        if self.analyze_convergence:
            x = (self.x + delta_x).detach()
            x.requires_grad_(True)

            # compute loss and gradient
            f = self.problem(x)
            loss = self.problem.ip_output(f, f)
            grad = TensorList(torch.autograd.grad(loss, x))

            # store in the vectors
            self.external_losses = torch.cat((self.external_losses, loss.detach().cpu().view(-1)))
            self.gradient_mags = torch.cat(
                (self.gradient_mags, sum(grad.view(-1) @ grad.view(-1)).cpu().sqrt().detach().view(-1)))
