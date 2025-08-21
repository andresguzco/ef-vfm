import torch.nn.functional as F
import torch
import numpy as np
from torchdiffeq import odeint_adjoint as odeint


class ExpVFM(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            vf_fn,
            device=torch.device('cpu'),
            **kwargs
        ):

        super(ExpVFM, self).__init__()

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device) if len(num_classes)>0 else torch.tensor([]).to(device).int()
        self.mask_index = torch.tensor(self.num_classes).long().to(device)
        self.neg_infinity = -1000000.0 
        self.num_classes_w_mask = tuple(self.num_classes + 1)

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = []
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(offsets).to(device)
        
        offsets = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes)+1)
        offsets = np.append([0], offsets)
        self.slices_for_classes_with_mask = []
        for i in range(1, len(offsets)):
            self.slices_for_classes_with_mask.append(np.arange(offsets[i - 1], offsets[i]))

        self._vf_fn = vf_fn
        self.device = device
    

    def mixed_loss(self, x):
        b = x.shape[0]
        dev = x.device

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:].long()

        t = torch.rand(b, device=dev, dtype=x_num.dtype)
        t = t[:, None]
            
        # Continuous interpolation
        x_num_t = x_num
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = t * x_num + (1 - t) * noise # + noise * sigma_num
        
        # Discrete interpolation
        x_cat_oh = self.to_one_hot(x_cat).float()
        if x_cat.shape[1] > 0:
            x_cat_t = t * x_cat_oh + (1 - t) * torch.randn_like(x_cat_oh)

        # Predict orignal data (distribution)
        model_out_num, model_out_cat = self._vf_fn(x_num_t, x_cat_t, t.squeeze())

        d_loss = torch.zeros((1,)).float()
        c_loss = torch.zeros((1,)).float()

        # Compute the loss
        if x_num.shape[1] > 0:
            c_loss = self._mvgloss(model_out_num, x_num, t)

        if x_cat.shape[1] > 0:
            d_loss = self._absorbed_closs(model_out_cat, x_cat, self._vf_fn.categories)
            
        return d_loss.mean(), c_loss.mean()
    
    def _mvgloss(self, mu_t, x_num_t, t):
        n, k = mu_t.shape
        dev = mu_t.device
        dt = mu_t.dtype

        identity = torch.eye(k, device=dev, dtype=dt).unsqueeze(0).expand(n, -1, -1)
        scale = 1 - (1 - 0.01) * t.unsqueeze(1) ** 2
        sigma = scale * identity
        dist = torch.distributions.MultivariateNormal(mu_t, sigma)
        return -dist.log_prob(x_num_t).mean()

    @torch.no_grad()
    def sample(self, num_samples):
        dev = self.device
        dt = torch.float32
        d_in = self.num_numerical_features + sum(self.num_classes) + len(self.num_classes)
        d_out = self.num_numerical_features + len(self.num_classes)

        x0 = torch.randn(num_samples, d_in, device=dev)
        t = torch.tensor([0.0, 0.999]).to(dev)
        vf = Velocity(self._vf_fn)
        trajectory = odeint(vf, x0, t, method="dopri5", rtol=1e-5, atol=1e-5)
        out = trajectory[1]

        sample = torch.zeros(num_samples, d_out, device=dev, dtype=dt)
        sample[:, :self.num_numerical_features] = out[:, :self.num_numerical_features].to(torch.float32)
        if sum(self.num_classes) != 0:
            idx = 0
            for i, val in enumerate(self.num_classes):
                sample[:, i] = torch.argmax(out[:, idx:idx + val], dim=1)
                idx += val
                assert val >= sample[:, i].max() >= 0, f"Sampled value {sample[:, i].max()} is out of range for categorical feature {i} with {val} classes."

        return sample.cpu()
    
    def sample_all(self, num_samples, batch_size, keep_nan_samples=False):        
        b = batch_size

        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            print(f"Samples left to generate: {num_samples-num_generated}")
            sample = self.sample(b)
            mask_nan = torch.any(sample.isnan(), dim=1)
            if keep_nan_samples:
                # If the sample instances that contains Nan are decided to be kept, the row with Nan will be foreced to all zeros
                sample = sample * (~mask_nan)[:, None]
            else:
                # Otherwise the instances with Nan will be eliminated
                sample = sample[~mask_nan]

            all_samples.append(sample)
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]

        return x_gen

    def to_one_hot(self, x_cat):
        
        x_cat_oh = torch.cat(   
            [F.one_hot(x_cat[:, i], num_classes=self.num_classes[i]) for i in range(len(self.num_classes))], 
            dim=-1
        )
        return x_cat_oh
    
    def _absorbed_closs(self, model_output, x0, cats): #, sigma, dsigma):
        """
            alpha: (bs,)
        """
        cum_sum =0
        losses = torch.zeros(len(cats), device=model_output.device)
        for i, val in enumerate(cats):
            dist = torch.distributions.Categorical(logits=model_output[:, cum_sum:cum_sum+val])
            losses[i] = -dist.log_prob(x0[:, i]).mean()
            cum_sum += val

        loss = losses.sum()
        return loss
    

class Velocity(torch.nn.Module):
    def __init__(self, model):
        super(Velocity, self).__init__()
        self.model = model

    def forward(self, t, x):
        t = t * torch.ones(x.shape[0]).to(x.device)

        x_num = x[:, :self.model.d_numerical]
        x_cat = x[:, self.model.d_numerical:]
        mu, logits = self.model(x_num, x_cat, t)

        pred = torch.cat([mu, logits], dim=1)
        v_t = (pred - (1 - 0.01) * x) / (1 - (1 - 0.01) * t.unsqueeze(1))
        return v_t