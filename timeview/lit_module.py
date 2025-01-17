import pytorch_lightning as pl
import torch
from .config import Config, OPTIMIZERS
from .model import TTS
import glob
import os
import pickle
import torch.nn.functional as F
#Lluis
from timeview.basis import BSplineBasis

from .hypermodel import HyperModel
def _get_seed_number(path):
    seeds = [os.path.basename(path).split("_")[1] for path in glob.glob(os.path.join(path, '*'))]
    seed = seeds[0]
    return seed

def _get_logs_seed_path(benchmarks_folder, timestamp, final=True, seed=None):

    # Create path
    if final:
        path = os.path.join(benchmarks_folder, timestamp, 'TTS', 'final', 'logs')
    else:
        path = os.path.join(benchmarks_folder, timestamp, 'TTS', 'tuning', 'logs')
    
    if seed is None:
        seed = _get_seed_number(path)

    logs_path = os.path.join(path, f'seed_{seed}')
    return logs_path

def _get_checkpoint_path_from_logs_seed_path(path):
    checkpoint_path = os.path.join(path, 'lightning_logs', 'version_0', 'checkpoints', 'best_val.ckpt')
    return checkpoint_path

def _load_config_from_logs_seed_path(path):
    config_path = os.path.join(path, 'config.pkl')
    # load config from a pickle
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    return config


def load_config(benchmarks_folder, timestamp, final=True, seed=None):

    logs_seed_path = _get_logs_seed_path(benchmarks_folder, timestamp, final=final, seed=seed)
    return _load_config_from_logs_seed_path(logs_seed_path)


def load_model(timestamp, benchmarks_folder='benchmarks', final=True, seed=None):

    logs_seed_path = _get_logs_seed_path(benchmarks_folder, timestamp, final=final, seed=seed)
    config = _load_config_from_logs_seed_path(logs_seed_path)
    checkpoint_path = _get_checkpoint_path_from_logs_seed_path(logs_seed_path)
    model = LitTTS.load_from_checkpoint(checkpoint_path, config=config)
    return model


class LitTTS(pl.LightningModule):

    def __init__(self, config: Config | None = None):
        super().__init__()
        if config is None:
            with open('config.pkl', 'rb') as f:
                config = pickle.load(f)
        self.config = config
        self.model = TTS(config)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = self.config.training.lr

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # def forward(self, batch, batch_idx, dataloader_idx=0):
        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)  # list of tensors
            return preds

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            return pred  # 2D tensor

    def training_step(self, batch, batch_idx):
        
        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]
            
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            losses = [self.loss_fn(pred, y)
                      for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))

        elif self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]

        self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](self.model.parameters(
        ), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)
        return optimizer
    
    def save_model(self, path='model.pth'):
        with open(os.path.join(os.path.dirname(path), 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path='model.pth'):
        self.model.load_state_dict(torch.load(path))

SPECIFIC = {
    'line_increasing': 0,
    'line_decreasing': 1,
    'line_constant': 2,
    'convex_increasing': 3,
    'concave_increasing': 4,
    'convex_decreasing': 5,
    'concave_decreasing': 6,
}

GENERAL = {
    0 : "increasing",
    1 : "decreasing",
    2 : "constant",
    3 : "increasing",
    4 : "increasing",
    5 : "decreasing",
    6 : "decreasing",
}


class LitHyperModel(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        with open('config.pkl', 'rb') as f:
            self.pretrained_config = pickle.load(f)
        self.pretrained_timeview = LitTTS(self.pretrained_config)
        self.pretrained_timeview.load_model("model.pth")
        self.pretrained_encoder = self.pretrained_timeview.model.encoder
        self.pretrained_encoder.eval()
        del self.pretrained_timeview.model
        del self.pretrained_timeview

        self.config = config
        self.model = HyperModel(config) 

        self.mse_fn = torch.nn.MSELoss()
        self.lr = self.config.training.lr if hasattr(self.config, 'training') else 1e-3
        self.weight_decay = (
            self.config.training.weight_decay if hasattr(self.config.training, 'weight_decay') else 0.0
        )
        self.optimizer_name = (
            self.config.training.optimizer if hasattr(self.config.training, 'optimizer') else 'adam'
        )
    def forward(self, X, Phis):
        return self.model(X, Phis)

    def forward_hyperbolic(self, X, timesteps):
        return self.model.forward_hyperbolic(X, timesteps)

    def compute_spline_loss(self, pred, batch_y, batch_X, batch_N):
        if self.config.dataloader_type == 'tensor':
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2),
                             dim=1) / batch_N) / batch_X.shape[0]
            return loss
        else:
            losses = []
            for p, t in zip(pred, batch_y):
                losses.append(self.mse_fn(p, t))
            return torch.mean(torch.stack(losses))
    
    def compute_aperture(self, point, K = 0.1, c = 1):
        #As we are using three equally spaced points, assume it's constant, for the Poincaré ball
        return 0.5

    def compositional_entailment_loss(self, parent_embeddings, child_embeddings, aperture=None):
        manifold = self.model.encoder.hyperbolic_manifold
        parent_embeddings = parent_embeddings.detach()
        if aperture is None:
            aperture = manifold.compute_entailment_cone(parent_embeddings[0])
        child_embeddings = child_embeddings.to(parent_embeddings.device)
        children = manifold.projx(child_embeddings)
        exterior_angles = manifold.compute_exterior_angle(parent_embeddings, children)
        losses = F.relu(exterior_angles - aperture)
        cel_loss = losses.mean()
        return cel_loss
    def compute_hyper_loss(self, hyper_out, true_motifs):
        embeddings = self.model.encoder.embeddings
        B, T, D = hyper_out.shape
        parent_list = []
        child_list = []
        for i in range(B):
            for j in range(T):
                child_id = true_motifs[i, j].item()
                parent_name = GENERAL[child_id]
                parent_vec = embeddings[parent_name]
                child_vec = hyper_out[i, j, :]

                parent_list.append(parent_vec)
                child_list.append(child_vec)

        if len(parent_list) == 0:
            print("No parent-child pairs found.")
            return torch.tensor(0.0, device=self.device)

    
        parent_stack = torch.stack(parent_list, dim=0)
        child_stack = torch.stack(child_list, dim=0)

        cel_loss = self.compositional_entailment_loss(parent_stack, child_stack, aperture=None)
        return cel_loss

    def compute_hyper_loss_iterative(self, hyper_out, target):
        embeddings = self.model.encoder.embeddings
        keys = [GENERAL[t] for t in target]
        parents = torch.stack([embeddings[k] for k in keys], dim=0)
        cel_loss = self.compositional_entailment_loss(parents, hyper_out)
        return cel_loss
    def pipeline_hyperbolic(self, X, true_data):
        list_of_x = []
        list_of_t = []
        target_list = []
        for i, (motif_ids_i, times_i) in enumerate(true_data):
            pair_list = []
            for j in range(len(motif_ids_i)):
                pair_list.append([times_i[j], times_i[j+1]]) 
            target_list = target_list + motif_ids_i
            timesteps_big = torch.tensor(pair_list, device=self.device)
            x_i = X[i]
            x_i_repeated = x_i.unsqueeze(0).expand(len(motif_ids_i), -1)  
            list_of_x.append(x_i_repeated)
            list_of_t.append(timesteps_big)
        x_big = torch.cat(list_of_x, dim=0)
        timesteps_big = torch.cat(list_of_t, dim=0)
        out = self.model.forward_hyperbolic(x_big, timesteps_big)  
        hyper_loss = self.compute_hyper_loss_iterative(out, target_list)
        return hyper_loss

    def training_step(self, batch, batch_idx):
        hyper_param = 0.1
        if self.config.dataloader_type == 'tensor':
            #batch_X, batch_Phi, batch_y, batch_NS, batch_TS = batch
            batch_X, batch_Phi, batch_y, batch_NS = batch
            #As TTS
            preds = self.model(batch_X, batch_Phi)  # shape (batch_size, #points)
            spline_loss = self.compute_spline_loss(preds, batch_y, batch_X, batch_NS)
            #Now our hyperbolic loss
            bspline = BSplineBasis(self.pretrained_config.n_basis, (0,self.pretrained_config.T), internal_knots=self.pretrained_config.internal_knots)
            true_motifs = self.get_motifs(bspline, batch_X)
            hyper_loss  = self.pipeline_hyperbolic(batch_X, true_motifs)
            loss = spline_loss + hyper_param * hyper_loss
        else:
            # iterative mode
            # parse (batch_X, batch_Phis, batch_ys)
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            spline_loss = self.compute_spline_loss(preds, batch_ys)

            hyper_loss = torch.tensor(0.0, device=self.device)  # or compute if you store timesteps
            loss = spline_loss + hyper_param * hyper_loss
        #print(f"train loss: {loss}, spline_loss: {spline_loss}, hyper_loss: {hyper_loss}")
        self.log("train_spline_loss", spline_loss)
        self.log("train_hyper_loss", hyper_loss)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        hyper_param = 0.0
        if self.config.dataloader_type == 'tensor':
            #batch_X, batch_Phi, batch_y, batch_NS, batch_TS = batch
            batch_X, batch_Phi, batch_y, batch_NS = batch
            preds = self.model(batch_X, batch_Phi)
            spline_loss = self.compute_spline_loss(preds, batch_y, batch_X, batch_NS)
            # bspline = BSplineBasis(self.pretrained_config.n_basis, (0,self.pretrained_config.T), internal_knots=self.pretrained_config.internal_knots)
            # true_motifs = self.get_motifs(bspline, batch_X)
            # hyper_loss  = self.pipeline_hyperbolic(batch_X, true_motifs)
            # loss = spline_loss + hyper_param * hyper_loss
            loss = spline_loss
        else:
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            loss = self.compute_spline_loss(preds, batch_ys)

        self.log("val_loss", loss)
        #print(f"val Loss: {loss}")
        return loss

    def test_step(self, batch, batch_idx):
        if self.config.dataloader_type == 'tensor':
            #batch_X, batch_Phi, batch_y, batch_NS, batch_TS = batch
            batch_X, batch_Phi, batch_y, batch_NS = batch
            preds = self.model(batch_X, batch_Phi)
            loss = self.compute_spline_loss(preds, batch_y, batch_X, batch_NS)
        else:
            batch_X, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Phis)
            loss = self.compute_spline_loss(preds, batch_ys)

        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        For forecasting. 
        """
        # If tensor
        if self.config.dataloader_type == 'tensor':
            batch_X, batch_Phi, _, _ = batch
            return self.model(batch_X, batch_Phi)
        else:
            batch_X, batch_Phis, _ = batch
            return self.model(batch_X, batch_Phis)

    def configure_optimizers(self):
        OptimClass = OPTIMIZERS[self.optimizer_name]
        optimizer = OptimClass(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def get_motifs(self, bspline, x):
        with torch.no_grad():
            c_torch = self.pretrained_encoder.forward(x)  
            if self.config.dynamic_bias:
                c_torch = c_torch[:,:-1]
        c_all = c_torch.cpu().numpy()
        motif_results = []
        for i in range(c_all.shape[0]):
            states, transitions = bspline.get_template_from_coeffs(c_all[i]) 
            # states => list of ints
            # transitions => list of times
            motif_results.append((states, transitions))
        return motif_results

    def on_after_backward(self):
        if self.model.encoder.hyperbolic_manifold.log_c.grad is not None:
            self.model.encoder.hyperbolic_manifold.log_c.grad.clamp_(0.1, 2)  # Clip gradients of log_c