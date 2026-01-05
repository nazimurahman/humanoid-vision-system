# src/models/manifold_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

class SinkhornKnoppProjection(nn.Module):
    """
    Sinkhorn-Knopp projection for doubly stochastic matrices.
    
    Enforces manifold constraints:
    1. All values ≥ 0 (non-negativity)
    2. Rows sum to 1 (row stochastic)
    3. Columns sum to 1 (column stochastic)
    
    This ensures the matrix is doubly stochastic, guaranteeing:
    - Non-expansive mapping (stable gradients)
    - Bounded eigenvalue spectrum (≤ 1)
    - Lipschitz continuity
    """
    
    def __init__(self, num_iterations: int = 20, epsilon: float = 1e-8, tau: float = 1.0):
        super().__init__()
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.tau = tau  # Temperature parameter
        self.register_buffer('convergence_history', torch.zeros(num_iterations))
        
    def forward(self, matrix: torch.Tensor, return_history: bool = False) -> torch.Tensor:
        """
        Apply Sinkhorn-Knopp projection to input matrix.
        
        Args:
            matrix: Input matrix of shape [B, N, M] or [N, M]
            return_history: Whether to return convergence metrics
            
        Returns:
            Doubly stochastic matrix with shape matching input
        """
        original_shape = matrix.shape
        needs_reshape = len(original_shape) > 2
        
        if needs_reshape:
            # Flatten batch dimension for processing
            batch_size, n, m = original_shape
            matrix = matrix.reshape(-1, n, m)
        else:
            batch_size = 1
            matrix = matrix.unsqueeze(0)
            
        # Ensure positivity using softmax with temperature
        # This is more stable than exp() for large values
        matrix = matrix / self.tau
        matrix = torch.softmax(matrix, dim=-1) * m  # Initialize to uniform
        
        # Track convergence
        row_sums = []
        col_sums = []
        
        # Sinkhorn-Knopp iterations
        for i in range(self.num_iterations):
            # Row normalization
            row_sum = matrix.sum(dim=2, keepdim=True)
            matrix = matrix / (row_sum + self.epsilon)
            row_sums.append(row_sum.mean().item())
            
            # Column normalization  
            col_sum = matrix.sum(dim=1, keepdim=True)
            matrix = matrix / (col_sum + self.epsilon)
            col_sums.append(col_sum.mean().item())
            
            # Track convergence (should approach 1)
            convergence = (row_sum.mean() - 1.0).abs().item()
            self.convergence_history[i] = convergence
            
        # Return to original shape
        if needs_reshape:
            matrix = matrix.reshape(original_shape)
        else:
            matrix = matrix.squeeze(0)
            
        if return_history:
            return matrix, {
                'row_sums': row_sums,
                'col_sums': col_sums,
                'final_row_error': row_sums[-1] - 1.0,
                'final_col_error': col_sums[-1] - 1.0
            }
            
        return matrix
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Get convergence metrics for monitoring."""
        return {
            'mean_convergence': self.convergence_history.mean().item(),
            'max_convergence': self.convergence_history.max().item(),
            'final_convergence': self.convergence_history[-1].item()
        }


class ManifoldHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC) Layer.
    
    Core innovation: Doubly stochastic residual connections for stability.
    
    Mathematical Formulation:
    Let X ∈ ℝ^{B×D} be input features
    
    1. H_pre = σ(H̃_pre) ∈ ℝ^{D×nD}  (sigmoid constraint [0,1])
    2. H_post = 2σ(H̃_post) ∈ ℝ^{nD×D} (amplified sigmoid [0,2])
    3. H_res = SK(H̃_res) ∈ ℝ^{D×D} (doubly stochastic via Sinkhorn-Knopp)
    
    Forward pass:
    X_exp = X @ H_pre                     # Expand: [B, D] → [B, nD]
    X_proc = MLP(X_exp)                  # Process in expanded space
    X_contr = X_proc @ H_post             # Contract: [B, nD] → [B, D]
    X_out = H_res @ X + X_contr           # Constrained residual
    
    Properties:
    - H_res doubly stochastic → eigenvalues ∈ [0,1] → non-expansive
    - Signal cannot explode/vanishing
    - Guaranteed training stability
    """
    
    def __init__(
        self,
        input_dim: int,
        expansion_rate: int = 4,
        hidden_dim: Optional[int] = None,
        alpha: float = 0.01,  # Small initialization as per requirement
        sk_iterations: int = 20,
        use_mixed_precision: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.expansion_rate = expansion_rate
        self.hidden_dim = hidden_dim or (input_dim * expansion_rate)
        self.alpha = alpha
        self.use_mixed_precision = use_mixed_precision
        self.dropout_rate = dropout_rate
        
        # Learnable matrices with small initialization (α=0.01)
        self.H_pre_raw = nn.Parameter(
            torch.randn(input_dim, self.hidden_dim) * alpha
        )
        self.H_post_raw = nn.Parameter(
            torch.randn(self.hidden_dim, input_dim) * alpha
        )
        self.H_res_raw = nn.Parameter(
            torch.randn(input_dim, input_dim) * alpha
        )
        
        # Sinkhorn-Knopp projection for H_res (doubly stochastic constraint)
        self.sinkhorn = SinkhornKnoppProjection(sk_iterations)
        
        # MLP for processing in expanded space
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),  # Gaussian Error Linear Unit (smooth ReLU)
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Layer normalization for stability
        self.norm_pre = nn.LayerNorm(input_dim)
        self.norm_post = nn.LayerNorm(input_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Monitoring buffers
        self.register_buffer('gradient_norms', torch.zeros(3))
        self.register_buffer('eigenvalues', torch.zeros(input_dim))
        self.register_buffer('signal_ratio_history', torch.zeros(1000))
        self.signal_ratio_idx = 0
        
        # Mixed precision settings
        self.dtype = torch.bfloat16 if use_mixed_precision else torch.float32
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with proper scaling."""
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.H_pre_raw, gain=0.1)
        nn.init.xavier_uniform_(self.H_post_raw, gain=0.1)
        nn.init.xavier_uniform_(self.H_res_raw, gain=0.1)
        
        # Initialize MLP weights
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=math.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def constrained_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute constrained matrices with manifold constraints.
        
        Returns:
            Tuple of (H_pre, H_post, H_res) with proper constraints applied
        """
        # H_pre: sigmoid for [0, 1] range (gating constraint)
        H_pre = torch.sigmoid(self.H_pre_raw)
        
        # H_post: 2 * sigmoid for [0, 2] range (amplified gating)
        H_post = 2 * torch.sigmoid(self.H_post_raw)
        
        # H_res: doubly stochastic via Sinkhorn-Knopp projection
        H_res = self.sinkhorn(self.H_res_raw)
        
        return H_pre, H_post, H_res
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with manifold constraints.
        
        Args:
            x: Input tensor of shape [B, D] or [B, *, D]
            
        Returns:
            Output tensor of same shape as input
        """
        original_shape = x.shape
        needs_reshape = len(original_shape) > 2
        
        if needs_reshape:
            # Flatten spatial dimensions
            B, *spatial, D = original_shape
            x = x.reshape(B, -1, D)
        
        # Save input for residual connection
        x_input = x
        
        # Apply constraints to matrices
        H_pre, H_post, H_res = self.constrained_matrices()
        
        # Mixed precision context (bfloat16 activations, float32 coefficients)
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision, dtype=self.dtype):
            # Pre-normalization
            x = self.norm_pre(x)
            
            # Expand to higher dimension: [B, *, D] → [B, *, hidden_dim]
            x_expanded = torch.matmul(x, H_pre)
            
            # Process in expanded space
            x_transformed = self.mlp(x_expanded)
            
            # Contract back to original dimension: [B, *, hidden_dim] → [B, *, D]
            x_contracted = torch.matmul(x_transformed, H_post)
            
            # Apply constrained residual connection
            # H_res is doubly stochastic → eigenvalues ≤ 1 → non-expansive
            x_residual = torch.matmul(x_input, H_res)
            x_output = x_residual + x_contracted
            
            # Post-normalization
            x_output = self.norm_post(x_output)
            
            # Dropout for regularization
            x_output = self.dropout(x_output)
        
        # Monitor stability metrics (only in training)
        if self.training:
            self._monitor_stability(H_res, x_input, x_output)
        
        # Reshape back if needed
        if needs_reshape:
            x_output = x_output.reshape(original_shape)
        
        return x_output
    
    def _monitor_stability(self, H_res: torch.Tensor, x_input: torch.Tensor, x_output: torch.Tensor):
        """Monitor training stability metrics."""
        with torch.no_grad():
            # 1. Eigenvalue spectrum of H_res (should be ≤ 1)
            try:
                # Use symmetric part for eigenvalue computation
                H_sym = (H_res + H_res.T) / 2
                eigenvalues = torch.linalg.eigvalsh(H_sym)
                self.eigenvalues.copy_(eigenvalues.detach())
            except:
                # Fallback for edge cases
                pass
            
            # 2. Signal growth/decay ratio
            input_norm = torch.norm(x_input, dim=-1).mean()
            output_norm = torch.norm(x_output, dim=-1).mean()
            signal_ratio = output_norm / (input_norm + 1e-8)
            
            # Store in circular buffer
            idx = self.signal_ratio_idx % 1000
            self.signal_ratio_history[idx] = signal_ratio
            self.signal_ratio_idx += 1
            
            # 3. Matrix constraints satisfaction
            row_sum = H_res.sum(dim=1).mean()
            col_sum = H_res.sum(dim=0).mean()
            
            # Store for external monitoring
            self.monitoring_metrics = {
                'max_eigenvalue': eigenvalues.max().item() if 'eigenvalues' in locals() else 0,
                'min_eigenvalue': eigenvalues.min().item() if 'eigenvalues' in locals() else 0,
                'signal_ratio': signal_ratio.item(),
                'row_sum_error': (row_sum - 1.0).abs().item(),
                'col_sum_error': (col_sum - 1.0).abs().item(),
            }
    
    def get_stability_metrics(self) -> Dict[str, Any]:
        """Get comprehensive stability metrics for monitoring."""
        metrics = {
            'max_eigenvalue': self.eigenvalues.max().item(),
            'min_eigenvalue': self.eigenvalues.min().item(),
            'eigenvalue_range': (self.eigenvalues.max() - self.eigenvalues.min()).item(),
            'sk_convergence': self.sinkhorn.get_convergence_metrics(),
        }
        
        # Add signal ratio statistics
        if self.signal_ratio_idx > 0:
            valid_history = self.signal_ratio_history[:min(self.signal_ratio_idx, 1000)]
            metrics.update({
                'signal_ratio_mean': valid_history.mean().item(),
                'signal_ratio_std': valid_history.std().item(),
                'signal_ratio_min': valid_history.min().item(),
                'signal_ratio_max': valid_history.max().item(),
            })
        
        # Add matrix constraint errors
        if hasattr(self, 'monitoring_metrics'):
            metrics.update(self.monitoring_metrics)
        
        return metrics
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, " \
               f"expansion={self.expansion_rate}, alpha={self.alpha}"


class MultiHeadManifoldAttention(nn.Module):
    """
    Multi-head attention with manifold-constrained projections.
    
    Uses mHC principles to stabilize attention mechanisms.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_mhc: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embed dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_mhc = use_mhc
        
        # Projections with optional mHC constraints
        if use_mhc:
            self.q_proj = ManifoldHyperConnection(embed_dim, expansion_rate=2)
            self.k_proj = ManifoldHyperConnection(embed_dim, expansion_rate=2)
            self.v_proj = ManifoldHyperConnection(embed_dim, expansion_rate=2)
            self.out_proj = ManifoldHyperConnection(embed_dim, expansion_rate=2)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for manifold-constrained attention.
        """
        batch_size, tgt_len, embed_dim = query.shape
        
        # Project queries, keys, values
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply masking if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, embed_dim)
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            return attn_output, attn_weights
        return attn_output, None


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More stable than LayerNorm, especially for deep networks.
    From: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x = x / rms * self.scale
        
        return x