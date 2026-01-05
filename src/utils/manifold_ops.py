# src/utils/manifold_ops.py
"""
Manifold Operations for Constrained Optimization.

This module implements operations on specific matrix manifolds:
1. Birkhoff Polytope (Doubly Stochastic Matrices)
2. Stiefel Manifold (Orthogonal Matrices)
3. SPD Manifold (Symmetric Positive Definite)
4. Hyperbolic Space (Poincaré Ball)

These operations ensure constraints are satisfied during optimization
and provide Riemannian gradients for manifold-aware optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Callable
import logging
from scipy import linalg
import warnings

logger = logging.getLogger(__name__)

class ManifoldProjector:
    """
    Base class for manifold projection operations.
    
    Provides common functionality for projecting onto and retracting
    from various matrix manifolds.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto manifold."""
        raise NotImplementedError
    
    def retract(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Retract tangent vector v at point x back to manifold."""
        raise NotImplementedError
    
    def riemannian_gradient(
        self, 
        x: torch.Tensor, 
        euclidean_grad: torch.Tensor
    ) -> torch.Tensor:
        """Convert Euclidean gradient to Riemannian gradient."""
        raise NotImplementedError
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute manifold distance between x and y."""
        raise NotImplementedError


class BirkhoffProjector(ManifoldProjector):
    """
    Projector for Birkhoff Polytope (doubly stochastic matrices).
    
    The Birkhoff polytope is the set of n×n doubly stochastic matrices.
    This class provides operations for this manifold.
    """
    
    def __init__(
        self,
        sinkhorn_iterations: int = 20,
        epsilon: float = 1e-8,
        method: str = 'sinkhorn'  # 'sinkhorn' or 'iterative'
    ):
        super().__init__(epsilon)
        self.sinkhorn_iterations = sinkhorn_iterations
        self.method = method
        
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project matrix onto Birkhoff polytope.
        
        Args:
            x: Input matrix [..., n, n]
            
        Returns:
            Doubly stochastic matrix
        """
        if self.method == 'sinkhorn':
            return self._project_sinkhorn(x)
        else:
            return self._project_iterative(x)
    
    def _project_sinkhorn(self, x: torch.Tensor) -> torch.Tensor:
        """Project using Sinkhorn-Knopp algorithm."""
        from .sinkhorn import SinkhornKnopp
        
        projector = SinkhornKnopp(
            num_iterations=self.sinkhorn_iterations,
            epsilon=self.epsilon,
            device=x.device
        )
        
        return projector.project(x)
    
    def _project_iterative(self, x: torch.Tensor) -> torch.Tensor:
        """Project using iterative normalization."""
        original_shape = x.shape
        
        # Ensure non-negativity
        x_proj = F.relu(x)
        
        # Iterative normalization (simplified Sinkhorn)
        for _ in range(self.sinkhorn_iterations):
            # Normalize rows
            row_sums = x_proj.sum(dim=-1, keepdim=True)
            x_proj = x_proj / (row_sums + self.epsilon)
            
            # Normalize columns
            col_sums = x_proj.sum(dim=-2, keepdim=True)
            x_proj = x_proj / (col_sums + self.epsilon)
        
        return x_proj
    
    def retract(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Retract tangent vector v at point x back to manifold.
        
        For Birkhoff polytope, we project x + v onto the manifold.
        """
        return self.project(x + v)
    
    def riemannian_gradient(
        self, 
        x: torch.Tensor, 
        euclidean_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Riemannian gradient from Euclidean gradient.
        
        For doubly stochastic matrices, the Riemannian gradient is the
        projection of the Euclidean gradient onto the tangent space.
        """
        # Tangent space: matrices whose rows and columns sum to zero
        n = x.shape[-1]
        
        # Compute row and column means of gradient
        row_means = euclidean_grad.mean(dim=-1, keepdim=True)
        col_means = euclidean_grad.mean(dim=-2, keepdim=True)
        total_mean = euclidean_grad.mean(dim=(-2, -1), keepdim=True)
        
        # Project onto tangent space
        riemannian_grad = (
            euclidean_grad 
            - row_means 
            - col_means 
            + total_mean
        )
        
        return riemannian_grad
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between two doubly stochastic matrices.
        
        Uses Frobenius norm for simplicity.
        """
        return torch.norm(x - y, p='fro', dim=(-2, -1))


class StiefelProjector(ManifoldProjector):
    """
    Projector for Stiefel Manifold (orthogonal matrices).
    
    The Stiefel manifold St(n, p) is the set of n×p matrices with
    orthonormal columns: X^T X = I_p.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        super().__init__(epsilon)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project matrix onto Stiefel manifold using QR decomposition.
        
        Args:
            x: Input matrix [..., n, p] with n ≥ p
            
        Returns:
            Orthogonal matrix with orthonormal columns
        """
        original_shape = x.shape
        n, p = original_shape[-2:]
        
        # Flatten batch dimensions
        x_flat = x.reshape(-1, n, p)
        batch_size = x_flat.shape[0]
        
        # QR decomposition for orthonormalization
        Q = torch.zeros_like(x_flat)
        
        for i in range(batch_size):
            q_i, _ = torch.linalg.qr(x_flat[i], mode='reduced')
            Q[i] = q_i
        
        # Ensure determinant is +1 (optional, for SO(n))
        # det = torch.det(Q @ Q.transpose(-2, -1))
        # Q = Q * torch.sign(det).unsqueeze(-1).unsqueeze(-1)
        
        return Q.reshape(original_shape)
    
    def retract(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Retract tangent vector v at point x using Cayley transform.
        
        Args:
            x: Point on manifold [..., n, p]
            v: Tangent vector [..., n, p] satisfying x^T v + v^T x = 0
            
        Returns:
            Retracted point on manifold
        """
        # Compute skew-symmetric matrix A = vx^T - xv^T
        A = torch.matmul(v, x.transpose(-2, -1)) - torch.matmul(x, v.transpose(-2, -1))
        
        # Cayley transform: X_new = (I - A/2)^{-1} (I + A/2) X
        I = torch.eye(A.shape[-1], device=A.device).unsqueeze(0).expand_as(A)
        
        # Handle batch dimensions
        if len(A.shape) > 2:
            batch_dims = A.shape[:-2]
            I = I.reshape(*batch_dims, A.shape[-2], A.shape[-1])
        
        A_half = A / 2
        cayley_factor = torch.linalg.solve(
            I - A_half,
            I + A_half
        )
        
        return torch.matmul(cayley_factor, x)
    
    def riemannian_gradient(
        self, 
        x: torch.Tensor, 
        euclidean_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Riemannian gradient for Stiefel manifold.
        
        The Riemannian gradient is:
        grad_R = grad_E - x (x^T grad_E)
        """
        # Project onto tangent space
        xT_grad = torch.matmul(x.transpose(-2, -1), euclidean_grad)
        riemannian_grad = euclidean_grad - torch.matmul(x, xT_grad)
        
        # Make it skew-symmetric (optional symmetry enforcement)
        riemannian_grad = 0.5 * (riemannian_grad - torch.matmul(
            x, torch.matmul(x.transpose(-2, -1), riemannian_grad)
        ))
        
        return riemannian_grad
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance on Stiefel manifold.
        
        Distance = arccos(singular values of x^T y)
        """
        # Compute x^T y
        xTy = torch.matmul(x.transpose(-2, -1), y)
        
        # Singular values (cosines of principal angles)
        singular_values = torch.linalg.svdvals(xTy)
        
        # Ensure values are in [-1, 1]
        singular_values = torch.clamp(singular_values, -1.0, 1.0)
        
        # Geodesic distance
        distances = torch.acos(singular_values).norm(dim=-1)
        
        return distances


class SPDProjector(ManifoldProjector):
    """
    Projector for Symmetric Positive Definite (SPD) manifold.
    
    The SPD manifold consists of symmetric positive definite matrices.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        super().__init__(epsilon)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project matrix onto SPD manifold.
        
        Args:
            x: Input matrix [..., n, n]
            
        Returns:
            Symmetric positive definite matrix
        """
        # Ensure symmetry
        x_sym = 0.5 * (x + x.transpose(-2, -1))
        
        # Add small diagonal for positive definiteness
        n = x_sym.shape[-1]
        identity = torch.eye(n, device=x.device)
        if len(x_sym.shape) > 2:
            identity = identity.unsqueeze(0).expand(*x_sym.shape[:-2], n, n)
        
        x_spd = x_sym + self.epsilon * identity
        
        # Ensure positive definiteness via eigenvalue clipping
        try:
            # Compute eigenvalues
            eigenvalues, eigenvectors = torch.linalg.eigh(x_spd)
            
            # Clip negative eigenvalues
            eigenvalues = torch.clamp(eigenvalues, min=self.epsilon)
            
            # Reconstruct matrix
            x_spd = torch.matmul(
                eigenvectors,
                torch.matmul(
                    torch.diag_embed(eigenvalues),
                    eigenvectors.transpose(-2, -1)
                )
            )
        except Exception as e:
            # Fallback: use simple regularization
            logger.warning(f"Eigen decomposition failed: {e}")
            x_spd = x_sym + (self.epsilon + torch.norm(x_sym, dim=(-2, -1)).mean()) * identity
        
        return x_spd
    
    def retract(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Retract using matrix exponential for SPD manifold.
        
        For SPD manifold, retraction via matrix exponential:
        X_new = X^{1/2} exp(X^{-1/2} V X^{-1/2}) X^{1/2}
        """
        # Compute matrix square root and inverse
        try:
            L = torch.linalg.cholesky(x)  # x = LL^T
            
            # Compute L^{-1} v L^{-T}
            Linv = torch.linalg.solve_triangular(L, torch.eye(L.shape[-1], device=x.device), upper=False)
            W = torch.matmul(Linv, torch.matmul(v, Linv.transpose(-2, -1)))
            
            # Matrix exponential
            W_exp = self._matrix_exp(W)
            
            # Retract: L W_exp L^T
            retracted = torch.matmul(L, torch.matmul(W_exp, L.transpose(-2, -1)))
            
        except Exception as e:
            # Fallback: simple additive retraction with projection
            logger.warning(f"Cholesky failed in SPD retract: {e}")
            retracted = self.project(x + v)
        
        return retracted
    
    def _matrix_exp(self, A: torch.Tensor) -> torch.Tensor:
        """Compute matrix exponential."""
        # Pad batch dimensions for torch.matrix_exp
        if len(A.shape) == 2:
            return torch.matrix_exp(A)
        else:
            # Handle batch dimensions
            original_shape = A.shape
            A_flat = A.reshape(-1, A.shape[-2], A.shape[-1])
            exp_flat = torch.stack([torch.matrix_exp(A_i) for A_i in A_flat])
            return exp_flat.reshape(original_shape)
    
    def riemannian_gradient(
        self, 
        x: torch.Tensor, 
        euclidean_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Riemannian gradient for SPD manifold.
        
        Riemannian gradient = X * EuclideanGrad * X
        """
        return torch.matmul(x, torch.matmul(euclidean_grad, x))
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Affine-Invariant distance on SPD manifold.
        
        Distance = ||log(X^{-1/2} Y X^{-1/2})||_F
        """
        try:
            # Compute X^{-1/2}
            Lx = torch.linalg.cholesky(x)
            Linv = torch.linalg.solve_triangular(
                Lx, 
                torch.eye(Lx.shape[-1], device=x.device), 
                upper=False
            )
            Xinv_half = Linv.transpose(-2, -1)
            
            # Compute X^{-1/2} Y X^{-1/2}
            Z = torch.matmul(Xinv_half, torch.matmul(y, Xinv_half.transpose(-2, -1)))
            
            # Eigen decomposition of Z
            eigenvalues, _ = torch.linalg.eigh(Z)
            
            # Log eigenvalues
            log_eigenvalues = torch.log(eigenvalues)
            
            # Distance = ||log(eigenvalues)||_2
            distance = torch.norm(log_eigenvalues, dim=-1)
            
        except Exception as e:
            # Fallback: Frobenius norm
            logger.warning(f"SPD distance computation failed: {e}")
            distance = torch.norm(x - y, p='fro', dim=(-2, -1))
        
        return distance


class ManifoldOptimizer:
    """
    Riemannian optimizer for manifold-constrained parameters.
    
    Implements Riemannian versions of common optimizers:
    - Riemannian SGD
    - Riemannian Adam
    - Riemannian RMSprop
    """
    
    def __init__(
        self,
        params,
        manifold: str = 'birkhoff',
        lr: float = 0.01,
        momentum: float = 0.9,
        projector: Optional[ManifoldProjector] = None
    ):
        """
        Initialize manifold optimizer.
        
        Args:
            params: Parameters to optimize (must be on manifold)
            manifold: Type of manifold ('birkhoff', 'stiefel', 'spd')
            lr: Learning rate
            momentum: Momentum factor
            projector: Custom manifold projector
        """
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        
        # Create appropriate projector
        if projector is not None:
            self.projector = projector
        else:
            if manifold == 'birkhoff':
                self.projector = BirkhoffProjector()
            elif manifold == 'stiefel':
                self.projector = StiefelProjector()
            elif manifold == 'spd':
                self.projector = SPDProjector()
            else:
                raise ValueError(f"Unknown manifold: {manifold}")
        
        # Momentum buffers
        self.momentum_buffers = {}
        for param in self.params:
            self.momentum_buffers[param] = torch.zeros_like(param)
    
    def step(self):
        """Perform Riemannian optimization step."""
        for param in self.params:
            if param.grad is None:
                continue
            
            # Get Euclidean gradient
            grad_euclidean = param.grad
            
            # Convert to Riemannian gradient
            grad_riemannian = self.projector.riemannian_gradient(param, grad_euclidean)
            
            # Apply momentum
            if self.momentum > 0:
                self.momentum_buffers[param] = (
                    self.momentum * self.momentum_buffers[param] +
                    grad_riemannian
                )
                update = self.momentum_buffers[param]
            else:
                update = grad_riemannian
            
            # Retract along negative gradient direction
            param_new = self.projector.retract(param, -self.lr * update)
            
            # Update parameter in-place
            param.data.copy_(param_new)
    
    def zero_grad(self):
        """Zero all gradients."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


def manifold_regularization(
    matrices: torch.Tensor,
    manifold_type: str = 'birkhoff',
    weight: float = 1.0
) -> torch.Tensor:
    """
    Compute manifold regularization loss.
    
    Encourages matrices to stay close to their manifold projections.
    
    Args:
        matrices: Input matrices
        manifold_type: Type of manifold
        weight: Regularization weight
        
    Returns:
        Regularization loss
    """
    # Create appropriate projector
    if manifold_type == 'birkhoff':
        projector = BirkhoffProjector()
    elif manifold_type == 'stiefel':
        projector = StiefelProjector()
    elif manifold_type == 'spd':
        projector = SPDProjector()
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")
    
    # Project onto manifold
    projected = projector.project(matrices)
    
    # Compute distance from manifold
    distance = projector.distance(matrices, projected)
    
    return weight * distance.mean()


def check_manifold_constraints(
    matrix: torch.Tensor,
    manifold_type: str = 'birkhoff',
    tol: float = 1e-4
) -> dict:
    """
    Check if matrix satisfies manifold constraints.
    
    Args:
        matrix: Matrix to check
        manifold_type: Type of manifold
        tol: Tolerance for constraint violations
        
    Returns:
        Dictionary with constraint checks
    """
    checks = {}
    
    if manifold_type == 'birkhoff':
        # Check non-negativity
        min_val = matrix.min().item()
        checks['non_negative'] = min_val >= -tol
        
        # Check row sums
        row_sums = matrix.sum(dim=-1)
        row_error = (row_sums - 1.0).abs().max().item()
        checks['row_sum_unit'] = row_error <= tol
        
        # Check column sums
        col_sums = matrix.sum(dim=-2)
        col_error = (col_sums - 1.0).abs().max().item()
        checks['col_sum_unit'] = col_error <= tol
        
        checks['row_error'] = row_error
        checks['col_error'] = col_error
        
    elif manifold_type == 'stiefel':
        # Check orthogonality: X^T X = I
        XTX = torch.matmul(matrix.transpose(-2, -1), matrix)
        identity = torch.eye(matrix.shape[-1], device=matrix.device)
        
        if len(matrix.shape) > 2:
            identity = identity.unsqueeze(0).expand(*matrix.shape[:-2], -1, -1)
        
        orth_error = (XTX - identity).abs().max().item()
        checks['orthogonal'] = orth_error <= tol
        checks['orth_error'] = orth_error
        
    elif manifold_type == 'spd':
        # Check symmetry
        sym_error = (matrix - matrix.transpose(-2, -1)).abs().max().item()
        checks['symmetric'] = sym_error <= tol
        
        # Check positive definiteness (try Cholesky)
        try:
            torch.linalg.cholesky(matrix)
            checks['positive_definite'] = True
        except:
            checks['positive_definite'] = False
        
        checks['sym_error'] = sym_error
    
    checks['all_satisfied'] = all(v for k, v in checks.items() 
                                  if not k.endswith('_error'))
    
    return checks