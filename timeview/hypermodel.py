import torch
import torch.nn as nn
from geoopt import PoincareBall
from .model import TTS
import math

class LearnablePoincareBall(nn.Module):
    def __init__(self, c_init=1.0, eps=1e-5):
        super(LearnablePoincareBall, self).__init__()
        self.log_c = nn.Parameter(torch.log(torch.tensor(c_init, dtype=torch.float32)))
        self.eps = torch.tensor(eps)

    @property
    def c(self):
        return torch.exp(self.log_c)

    def projx(self, x):
        norm_x = torch.norm(x, dim=-1, keepdim=True).to(self.c.device)
        radius = 1.0 / torch.sqrt(self.c)
        cond = norm_x >= (radius - self.eps.to(self.c.device))
        if not torch.any(cond):
            return x
        scale = (radius - self.eps) / (norm_x + self.eps)
        x_proj = x * scale
        return torch.where(cond, x_proj, x)
    def exp_map_zero(self, v):
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        scale = torch.tanh(math.sqrt(self.c) * norm_v) / (math.sqrt(self.c) * (norm_v + self.eps))
        return v * scale
    def log_map_zero(self, x):
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        scale = (1.0 / math.sqrt(self.c)) * torch.atanh(math.sqrt(self.c) * norm_x) / (norm_x + self.eps)
        return x * scale

    def mobius_add(self, x, y):
        c = self.c
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + c**2 * torch.sum(y * y, dim=-1, keepdim=True) * torch.sum(x * x, dim=-1, keepdim=True)
        return numerator / (denominator + self.eps)
    
    def distance(self, x, y):
        return torch.acosh(1 + 2 * torch.sum((x - y) ** 2, dim=-1) / ((1 - torch.sum(x ** 2, dim=-1)) * (1 - torch.sum(y ** 2, dim=-1)) + self.eps))
    
class LearnableLorentzModel(nn.Module):
    def __init__(self, c_init=1.0, eps=1e-5, dim=5):
        super(LearnableLorentzModel, self).__init__()
        self.log_c = nn.Parameter(torch.log(torch.tensor(c_init, dtype=torch.float32)))
        self.eps = torch.tensor(eps)
        self.dim = dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def c(self):
        clamped_log_c = torch.clamp(self.log_c, min=-10, max=10) 
        return torch.exp(clamped_log_c).to(self.device)

    def lorentz_inner_product(self, p, q):
        return -p[..., 0] * q[..., 0] + torch.sum(p[..., 1:] * q[..., 1:], dim=-1)

    def lorentz_distance(self, p, q):
        lpq = self.lorentz_inner_product(p, q)
        return torch.acosh(-lpq / self.c + self.eps)

    def projx(self, x):
        x = x.clone().to(self.device)
        norm_spatial = torch.norm(x[..., 1:], dim=-1, keepdim=True).to(self.device)
        updated_x0 = torch.sqrt(1.0 / self.c + norm_spatial**2)
        x = torch.cat((updated_x0, x[..., 1:]), dim=-1) 
        return x

    def exp_map_zero(self, v):
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        scale = torch.sqrt(self.c) * torch.cosh(norm_v * torch.sqrt(self.c)) / (norm_v + self.eps)
        p = torch.zeros_like(v)
        p[..., 0] = scale
        p[..., 1:] = v * torch.tanh(norm_v * torch.sqrt(self.c)) / (norm_v + self.eps)
        return p

    def log_map_zero(self, p):
        norm_spatial = torch.norm(p[..., 1:], dim=-1, keepdim=True)
        scale = (1.0 / torch.sqrt(self.c)) * torch.acosh(p[..., 0] * torch.sqrt(self.c)) / (norm_spatial + self.eps)
        v = p[..., 1:] * scale
        return v

    def compute_entailment_cone(self, general, eta=1.0):
        general_norm = torch.norm(general[..., 1:], dim=-1, keepdim=True)
        half_aperture = eta * torch.asin(2 * torch.sqrt(self.c) * general_norm)
        return half_aperture
    
    def compute_exterior_angle(self, p, q):
        numerator = p[..., 0] + q[..., 0] * self.c * self.lorentz_inner_product(p, q)

        lorentz_ip = self.lorentz_inner_product(p, q)
        sqrt_term = torch.sqrt(torch.clamp((self.c * lorentz_ip)**2 - 1, min=0.0))  # Clamp to avoid sqrt of negative
        denominator = torch.norm(q[..., 1:], dim=-1, keepdim=True) * sqrt_term

        ratio = numerator / (denominator + self.eps)
        ratio = torch.clamp(ratio, -1.0 + self.eps.to(self.device), 1.0 - self.eps.to(self.device))

        return torch.acos(ratio)

class MultiHeadNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_features = config.n_features
        self.n_basis = config.n_basis
        self.hidden_sizes = config.encoder.hidden_sizes
        self.dropout_p = config.encoder.dropout_p
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.hyperbolic_dim = 5 # hardcoded
        self.first_layer = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_sizes[0]),
            nn.BatchNorm1d(self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )
        self.main_layers = nn.ModuleList()
        for i in range(len(self.hidden_sizes) - 2):
            self.main_layers.append(nn.Sequential(
                nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]),
                nn.BatchNorm1d(self.hidden_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
            ))
        self.spline_layer = nn.Sequential(
            nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1]),
            nn.BatchNorm1d(self.hidden_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )
        latent_size = self.n_basis
        if self.is_dynamic_bias_enabled(config):
            latent_size += 1

        self.hyper_layer = nn.Sequential(
            nn.Linear(self.hidden_sizes[-2] + 2, self.hidden_sizes[-1]),
            nn.BatchNorm1d(self.hidden_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )
        
        
        self.spline_head = nn.Linear(self.hidden_sizes[-1], latent_size)
    
        self.hyperbolic_head = nn.Linear(self.hidden_sizes[-1], config.hyperbolic_dim)
        self.hyperbolic_manifold = LearnableLorentzModel(dim=config.hyperbolic_dim).to(self.device)
        self.embeddings = {}
        self.general_motifs = ["increasing", "decreasing", "constant"]
        basis_vectors = torch.eye(config.hyperbolic_dim)[:len(self.general_motifs)]
        radius = 0.5 # hardcoded
        for i, motif in enumerate(self.general_motifs):
            angle = 2 * math.pi * i / len(self.general_motifs)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vec = torch.zeros(config.hyperbolic_dim)
            assert config.hyperbolic_dim >= 3
            vec[1] = x
            vec[2] = y
            self.embeddings[motif] = self.hyperbolic_manifold.projx(vec)

    @staticmethod
    def is_dynamic_bias_enabled(config):
        return hasattr(config, 'dynamic_bias') and config.dynamic_bias

    def encode_spline(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        for layer in self.main_layers:
            x = layer(x)
        x = self.spline_layer(x)
        return x

    def encode_hyper(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        for i, layer in enumerate(self.main_layers):
            x = layer(x)
        if timesteps.dtype != torch.float32:
            timesteps = timesteps.to(torch.float32)
        x = torch.cat([x, timesteps.to(x.device)], dim=1)
        return x

    def forward_spline(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        hidden = self.encode_spline(x)
        spline_output = self.spline_head(hidden)
        return spline_output

    def forward_hyperbolic(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        timesteps = timesteps.to(self.device)
        hidden = self.encode_hyper(x, timesteps)
        hidden = self.hyper_layer(hidden)
        hyper_out = self.hyperbolic_head(hidden)
        hyper_out = self.hyperbolic_manifold.projx(hyper_out)  # project onto manifold
        return hyper_out

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor): #Not used
        s = self.forward_spline(x)
        h = self.forward_hyperbolic(x, timesteps)
        return s, h



class HyperModel(TTS):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = MultiHeadNetwork(config)

        if not self.is_dynamic_bias_enabled(self.config):
            self.bias = torch.nn.Parameter(torch.zeros(1))
        
    @staticmethod
    def is_dynamic_bias_enabled(config):
        return hasattr(config, 'dynamic_bias') and config.dynamic_bias

    def forward(self, X, Phis):
        spline_output = self.encoder.forward_spline(X)
        
        if self.is_dynamic_bias_enabled(self.config):
            dynamic_bias = spline_output[:, -1]
            spline_output = spline_output[:, :-1]
        else:
            dynamic_bias = self.bias

        if self.config.dataloader_type == "iterative":
            results = []
            for d, Phi in enumerate(Phis):
                if self.is_dynamic_bias_enabled(self.config):
                    # shape: ( #points, ) after matmul
                    out = torch.matmul(Phi, spline_output[d, :]) + dynamic_bias[d]
                else:
                    out = torch.matmul(Phi, spline_output[d, :]) + self.bias
                results.append(out)
            return results

        elif self.config.dataloader_type == "tensor":
            if self.is_dynamic_bias_enabled(self.config):
                return torch.matmul(Phis,torch.unsqueeze(spline_output,-1)).squeeze(-1) + torch.unsqueeze(dynamic_bias,-1)
            else:
                return torch.matmul(Phis,torch.unsqueeze(spline_output,-1)).squeeze(-1) + dynamic_bias
            

    def forward_hyperbolic(self, X, timesteps):
        hyper_out = self.encoder.forward_hyperbolic(X, timesteps)
        return hyper_out