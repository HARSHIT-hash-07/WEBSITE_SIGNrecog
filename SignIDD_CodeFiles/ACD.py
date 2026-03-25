# coding: utf-8
import math
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F

from collections import namedtuple
from torch import nn
from ACD_Denoiser import ACD_Denoiser
from ID import ID

__all__ = ["ACD"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class ACD(nn.Module):

    def __init__(self, args, trg_vocab):
        super().__init__()

        timesteps = args["diffusion"].get('timesteps', 1000)
        sampling_timesteps = args["diffusion"].get('sampling_timesteps', 5)

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = args["diffusion"].get('scale', 1.0)
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # Above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)

        # Below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.ACD_Denoiser = ACD_Denoiser(num_layers=args["diffusion"].get('num_layers', 2),
                                         num_heads=args["diffusion"].get('num_heads', 4),
                                         hidden_size=args["diffusion"].get('hidden_size', 512),
                                         ff_size=args["diffusion"].get('ff_size', 512),
                                         dropout=args["diffusion"].get('dropout', 0.1),
                                         emb_dropout=args["diffusion"].get("embeddings", {}).get('dropout', 0.1),
                                         vocab_size=len(trg_vocab),
                                         freeze=False,
                                         trg_size=args.get('trg_size', 150),
                                         decoder_trg_trg_=True)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, encoder_output, t, src_mask, trg_mask):
        x_t = ID(x)
        x_t = x_t / self.scale

        pred_pose = self.ACD_Denoiser(encoder_output=encoder_output,
                                      trg_embed=x_t,
                                      src_mask=src_mask,
                                      trg_mask=trg_mask,
                                      t=t)

        x_start = pred_pose
        x_start = x_start * self.scale
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def ddim_sample(self, encoder_output, input_3d, src_mask, trg_mask):
        batch = encoder_output.shape[0]
        shape = (batch, input_3d.shape[1], 150)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        preds_all=[]
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            preds = self.model_predictions(x=img, encoder_output=encoder_output, t=time_cond,src_mask=src_mask, trg_mask=trg_mask)
            pred_noise, x_start = preds.pred_noise.float(), preds.pred_x_start
            preds_all.append(x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return preds_all

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, encoder_output, input_3d, src_mask, trg_mask, is_train):

        # Prepare Proposals.
        if not is_train:
            results = self.ddim_sample(encoder_output=encoder_output, input_3d=input_3d, src_mask=src_mask, trg_mask=trg_mask)
            return results[self.sampling_timesteps-1]

        if is_train:
            x_poses, noises, t = self.prepare_targets(input_3d)
            x_poses = x_poses.float()
            x_poses = ID(x_poses)
            t = t.squeeze(-1)
            pred_pose = self.ACD_Denoiser(encoder_output=encoder_output,
                                          trg_embed=x_poses,
                                          src_mask=src_mask,
                                          trg_mask=trg_mask,
                                          t=t)
            return pred_pose

    def prepare_diffusion_concat(self, pose_3d):

        t = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        noise = torch.randn(pose_3d.shape[0],150, device=device)

        x_start = pose_3d

        x_start = x_start * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = x / self.scale

        return x, noise, t

    def prepare_targets(self, targets):
        diffused_poses = []
        noises = []
        ts = []
        for i in range(0,targets.shape[0]):
            targets_per_sample = targets[i]

            d_poses, d_noise, d_t = self.prepare_diffusion_concat(targets_per_sample)
            diffused_poses.append(d_poses)
            noises.append(d_noise)
            ts.append(d_t)

        return torch.stack(diffused_poses), torch.stack(noises), torch.stack(ts)









# # Ankita mam code
# # coding: utf-8
# import math
# import torch
# import torch.nn.functional as F
# from collections import namedtuple
# from torch import nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from ACD_Denoiser import ACD_Denoiser
# from ID import ID
# from helpers import make_joint_channel_masks  # <-- needs to exist

# __all__ = ["ACD"]

# ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# def exists(x):
#     return x is not None


# def default(val, d):
#     if exists(val):
#         return val
#     return d() if callable(d) else d


# def extract(a, t, x_shape):
#     """
#     extract the appropriate t index for a batch of indices
#     a: [T]
#     t: [B] (long)
#     return: [B, 1, 1, ...] broadcasting shape to x_shape
#     """
#     batch_size = t.shape[0]
#     out = a.gather(-1, t)  # [B]
#     return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# def cosine_beta_schedule(timesteps, s=0.008):
#     """Cosine schedule (https://openreview.net/forum?id=-NEXDKk8gZ)"""
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0, 0.999).float()


# class ACD(nn.Module):

#     def __init__(self, args, trg_vocab):
#         super().__init__()

#         # -------- Schedules (two-rate: Body vs Hand) --------
#         timesteps = args["diffusion"].get('timesteps', 1000)
#         sampling_timesteps = args["diffusion"].get('sampling_timesteps', 5)
#         hand_beta_scale = args["diffusion"].get('hand_beta_scale', 0.6)  # <1 => hands corrupted slower

#         base_betas = cosine_beta_schedule(timesteps)       # Body schedule
#         betas_B = base_betas
#         betas_H = (base_betas * hand_beta_scale).clamp(max=0.999)

#         def cumprod_stats(betas):
#             alphas = 1. - betas
#             ac = torch.cumprod(alphas, dim=0)  # ᾱ_t
#             return {
#                 "betas": betas,
#                 "alphas_cumprod": ac,
#                 "sqrt_ac": torch.sqrt(ac),
#                 "sqrt_1m_ac": torch.sqrt(1. - ac),
#                 "sqrt_recip_ac": torch.sqrt(1. / ac),
#                 "sqrt_recipm1_ac": torch.sqrt(1. / ac - 1),
#             }

#         stats_B = cumprod_stats(betas_B)
#         stats_H = cumprod_stats(betas_H)

#         # Register buffers (Body: *_B, Hand: *_H)
#         for k, v in stats_B.items():
#             self.register_buffer(f"{k}_B", v)
#         for k, v in stats_H.items():
#             self.register_buffer(f"{k}_H", v)

#         # Keep some of the original single-schedule artifacts (not strictly required for DDIM)
#         # but harmless to have around
#         self.num_timesteps = int(betas_B.shape[0])
#         self.sampling_timesteps = default(sampling_timesteps, self.num_timesteps)
#         assert self.sampling_timesteps <= self.num_timesteps
#         self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
#         self.ddim_sampling_eta = 1.0

#         # misc flags from original
#         self.self_condition = False
#         self.scale = args["diffusion"].get('scale', 1.0)
#         self.box_renewal = True
#         self.use_ensemble = True

#         # -------- Denoiser --------
#         self.ACD_Denoiser = ACD_Denoiser(
#             num_layers=args["diffusion"].get('num_layers', 2),
#             num_heads=args["diffusion"].get('num_heads', 4),
#             hidden_size=args["diffusion"].get('hidden_size', 512),
#             ff_size=args["diffusion"].get('ff_size', 512),
#             dropout=args["diffusion"].get('dropout', 0.1),
#             emb_dropout=args["diffusion"]["embeddings"].get('dropout', 0.1),
#             vocab_size=len(trg_vocab),
#             freeze=False,
#             trg_size=args.get('trg_size', 150),
#             decoder_trg_trg_=True
#         )

#     # ---------- Group-aware helpers ----------

#     def _sigmas_pair(self, t, x_shape):
#         """
#         Return σ_t^B, σ_t^H as [B] scalars:
#           σ_t^C := 1 - ᾱ_t^C
#         """
#         acB = extract(self.alphas_cumprod_B, t, x_shape).squeeze(-1).squeeze(-1)  # [B]
#         acH = extract(self.alphas_cumprod_H, t, x_shape).squeeze(-1).squeeze(-1)  # [B]
#         return (1. - acB).float(), (1. - acH).float()

#     def predict_noise_from_start_grouped(self, x_t, t, x0):
#         """
#         ε̂ = (sqrt(1/ᾱ_t) * x_t - x0) / sqrt(1/ᾱ_t - 1)
#         but mixed per-channel using hand/body masks.
#         """
#         mask_body, mask_hand = make_joint_channel_masks(device=x_t.device)  # [1,1,150]

#         sr_B = extract(self.sqrt_recip_ac_B, t, x_t.shape)       # [B,1,1]
#         srm1_B = extract(self.sqrt_recipm1_ac_B, t, x_t.shape)
#         sr_H = extract(self.sqrt_recip_ac_H, t, x_t.shape)
#         srm1_H = extract(self.sqrt_recipm1_ac_H, t, x_t.shape)

#         sr = sr_B * (~mask_hand) + sr_H * mask_hand              # [B,1,150]
#         srm1 = srm1_B * (~mask_hand) + srm1_H * mask_hand

#         return (sr * x_t - x0) / srm1

#     # ---------- Core model calls ----------

#     def model_predictions(self, x, encoder_output, t, src_mask, trg_mask):
#         """
#         Given current noisy sample x (B,T,150), return:
#           - pred_noise (ε̂)
#           - x_start (x̂₀)
#         """
#         x_t = ID(x)  # (B,T,50*7) iconicity / dir+len expansion
#         x_t = x_t / self.scale

#         # Optional: condition denoiser with σ^B, σ^H
#         sigma_B, sigma_H = self._sigmas_pair(t, x.shape)  # [B], [B]

#         # Call denoiser; pass σ if its forward supports it
#         try:
#             pred_pose = self.ACD_Denoiser(
#                 encoder_output=encoder_output,
#                 trg_embed=x_t,
#                 src_mask=src_mask,
#                 trg_mask=trg_mask,
#                 t=t,
#                 sigma_B=sigma_B,
#                 sigma_H=sigma_H
#             )
#         except TypeError:
#             # Backward-compatible: older denoiser without sigma args
#             pred_pose = self.ACD_Denoiser(
#                 encoder_output=encoder_output,
#                 trg_embed=x_t,
#                 src_mask=src_mask,
#                 trg_mask=trg_mask,
#                 t=t
#             )

#         x_start = pred_pose * self.scale
#         pred_noise = self.predict_noise_from_start_grouped(x, t, x_start)

#         return ModelPrediction(pred_noise, x_start)

#     # ---------- Sampling ----------

#     def ddim_sample(self, encoder_output, input_3d, src_mask, trg_mask):
#         """
#         DDIM sampling with group-wise mixed alphas for Body vs Hands.
#         """
#         batch = encoder_output.shape[0]
#         shape = (batch, input_3d.shape[1], 150)
#         total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

#         # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
#         times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
#         times = list(reversed(times.int().tolist()))
#         time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), ..., (0, -1)]

#         img = torch.randn(shape, device=device)

#         x_start = None
#         preds_all = []
#         mask_body, mask_hand = make_joint_channel_masks(device=img.device)

#         for time, time_next in time_pairs:
#             time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

#             preds = self.model_predictions(
#                 x=img, encoder_output=encoder_output, t=time_cond,
#                 src_mask=src_mask, trg_mask=trg_mask
#             )
#             pred_noise, x_start = preds.pred_noise.float(), preds.pred_x_start
#             preds_all.append(x_start)

#             if time_next < 0:
#                 img = x_start
#                 continue

#             # Per-group alphas
#             alpha_B = self.alphas_cumprod_B[time]
#             alpha_n_B = self.alphas_cumprod_B[time_next]
#             alpha_H = self.alphas_cumprod_H[time]
#             alpha_n_H = self.alphas_cumprod_H[time_next]

#             sigma_B = eta * ((1 - alpha_B / alpha_n_B) * (1 - alpha_n_B) / (1 - alpha_B)).sqrt()
#             sigma_H = eta * ((1 - alpha_H / alpha_n_H) * (1 - alpha_n_H) / (1 - alpha_H)).sqrt()
#             c_B = (1 - alpha_n_B - sigma_B ** 2).sqrt()
#             c_H = (1 - alpha_n_H - sigma_H ** 2).sqrt()

#             # Mix per channel
#             alpha_n_mix = alpha_n_B.sqrt() * (~mask_hand) + alpha_n_H.sqrt() * mask_hand  # [1,1,150]
#             c_mix = c_B * (~mask_hand) + c_H * mask_hand
#             sigma_mix = sigma_B * (~mask_hand) + sigma_H * mask_hand

#             noise = torch.randn_like(img)
#             img = x_start * alpha_n_mix + c_mix * pred_noise + sigma_mix * noise

#         return preds_all

#     # ---------- Training-time forward noising ----------

#     # def q_sample(self, x_start, t, noise=None):
#     #     """
#     #     Group-aware forward diffusion:
#     #       x_t[C] = sqrt(ᾱ_t^C) * x_0[C] + sqrt(1-ᾱ_t^C) * ε[C]
#     #     where C in {Body, Hand}.
#     #     """
#     #     if noise is None:
#     #         noise = torch.randn_like(x_start)

#     #     mask_body, mask_hand = make_joint_channel_masks(device=x_start.device)  # [1,1,150]

#     #     sqrt_ac_B = extract(self.sqrt_ac_B, t, x_start.shape)
#     #     sqrt_1m_B = extract(self.sqrt_1m_ac_B, t, x_start.shape)
#     #     sqrt_ac_H = extract(self.sqrt_ac_H, t, x_start.shape)
#     #     sqrt_1m_H = extract(self.sqrt_1m_ac_H, t, x_start.shape)

#     #     sqrt_ac = sqrt_ac_B * (~mask_hand) + sqrt_ac_H * mask_hand      # [B,1,150]
#     #     sqrt_1m = sqrt_1m_B * (~mask_hand) + sqrt_1m_H * mask_hand

#     #     return sqrt_ac * x_start + sqrt_1m * noise

#     def q_sample(self, x_start, t, noise=None):
#         """
#         Group-aware forward diffusion for TRAINING (x_start: [T,150]).
#         Mix Body/Hand scalars over channels -> [1,150], so broadcasting keeps shape [T,150].
#         """
#         if noise is None:
#             noise = torch.randn_like(x_start)
    
#         device = x_start.device
    
#         # masks: originally [1,1,150] -> squeeze time axis to [1,150]
#         _, mask_hand_150 = make_joint_channel_masks(device=device)   # [1,1,150] (bool)
#         mask_hand = mask_hand_150.squeeze(1).to(dtype=x_start.dtype)  # [1,150] (float)
#         mask_body = 1.0 - mask_hand                                   # [1,150]
    
#         # per-step scalars come out as [1,1]; let them broadcast to [1,150]
#         sqrt_ac_B  = extract(self.sqrt_ac_B,    t, x_start.shape)     # [1,1]
#         sqrt_1m_B  = extract(self.sqrt_1m_ac_B, t, x_start.shape)     # [1,1]
#         sqrt_ac_H  = extract(self.sqrt_ac_H,    t, x_start.shape)     # [1,1]
#         sqrt_1m_H  = extract(self.sqrt_1m_ac_H, t, x_start.shape)     # [1,1]
    
#         # mix to per-channel [1,150], then broadcast with [T,150] -> [T,150]
#         sqrt_ac = sqrt_ac_B * mask_body + sqrt_ac_H * mask_hand       # [1,150]
#         sqrt_1m = sqrt_1m_B * mask_body + sqrt_1m_H * mask_hand       # [1,150]
    
#         return sqrt_ac * x_start + sqrt_1m * noise                    # [T,150]

    

#     # ---------- Top-level forward ----------

#     def forward(self, encoder_output, input_3d, src_mask, trg_mask, is_train):
#         # Inference: return last x̂₀ from DDIM
#         if not is_train:
#             results = self.ddim_sample(
#                 encoder_output=encoder_output, input_3d=input_3d,
#                 src_mask=src_mask, trg_mask=trg_mask
#             )
#             return results[self.sampling_timesteps - 1]

#         # Training: sample t and noise target poses, predict x̂₀
#         x_poses, noises, t = self.prepare_targets(input_3d)  # x_t, ε, t
#         x_poses = x_poses.float()
#         x_poses = ID(x_poses)  # iconicity expansion (B,T,50*7)
#         t = t.squeeze(-1)

#         # Optional σ conditioning (same as in model_predictions)
#         sigma_B, sigma_H = self._sigmas_pair(t, x_poses.shape)
#         try:
#             pred_pose = self.ACD_Denoiser(
#                 encoder_output=encoder_output,
#                 trg_embed=x_poses,
#                 src_mask=src_mask,
#                 trg_mask=trg_mask,
#                 t=t,
#                 sigma_B=sigma_B,
#                 sigma_H=sigma_H
#             )
#         except TypeError:
#             pred_pose = self.ACD_Denoiser(
#                 encoder_output=encoder_output,
#                 trg_embed=x_poses,
#                 src_mask=src_mask,
#                 trg_mask=trg_mask,
#                 t=t
#             )
#         return pred_pose

#     # ---------- Target prep (unchanged API, but uses group-aware q_sample) ----------

#     def prepare_diffusion_concat(self, pose_3d):
#         """
#         Sample a single t and produce (x_t, noise, t) for one sample sequence.
#         """
#         t = torch.randint(0, self.num_timesteps, (1,), device=device).long()
#         noise = torch.randn(pose_3d.shape[0], 150, device=device)

#         x_start = pose_3d * self.scale
#         x = self.q_sample(x_start=x_start, t=t, noise=noise)  # group-aware
#         x = x / self.scale

#         return x, noise, t

#     def prepare_targets(self, targets):
#         """
#         For a batch of sequences: return stacked x_t, ε, t.
#         """
#         diffused_poses = []
#         noises = []
#         ts = []
#         for i in range(0, targets.shape[0]):
#             targets_per_sample = targets[i]  # [T,150]
#             d_poses, d_noise, d_t = self.prepare_diffusion_concat(targets_per_sample)
#             diffused_poses.append(d_poses)
#             noises.append(d_noise)
#             ts.append(d_t)

#         return torch.stack(diffused_poses), torch.stack(noises), torch.stack(ts)
