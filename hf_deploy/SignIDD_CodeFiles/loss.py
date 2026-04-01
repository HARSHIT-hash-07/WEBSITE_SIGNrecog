# coding: utf-8
import torch
import torch.nn as nn
from typing import Optional

from helpers import getSkeletalModelStructure, getSkeletalParentsDict
from pinn_losses import PINNLoss, PINNConfig

class Loss(nn.Module):

    def __init__(self, cfg, target_pad=0.0):
        super(Loss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()
        self.bone_loss = cfg["training"]["bone_loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        if self.bone_loss == "l1":
            self.criterion_bone = nn.L1Loss()
        elif self.bone_loss == "mse":
            self.criterion_bone = nn.MSELoss()
        else:
            print("Loss not found - revert to default MSE loss")
            self.criterion_bone = nn.MSELoss()

        model_cfg = cfg["model"]
        training_cfg = cfg["training"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

        # PINN Configuration
        self.use_pinn = training_cfg.get("use_pinn", False)
        self.lambda_pinn = float(training_cfg.get("lambda_pinn", 0.5))
        
        if self.use_pinn:
            # Create PINN configuration from training config with proper type conversion
            pinn_cfg = PINNConfig(
                lambda_bone=float(training_cfg.get("pinn_lambda_bone", 1.0)),
                lambda_vel=float(training_cfg.get("pinn_lambda_vel", 0.1)),
                lambda_acc=float(training_cfg.get("pinn_lambda_acc", 0.05)),
                lambda_fk=float(training_cfg.get("pinn_lambda_fk", 0.5)),
                eps=float(training_cfg.get("pinn_eps", 1e-8)),
                dt=float(training_cfg.get("pinn_dt", 1.0)),
                rest_from=str(training_cfg.get("pinn_rest_from", "first_valid")),
                detach_rest=bool(training_cfg.get("pinn_detach_rest", True)),
                use_huber=bool(training_cfg.get("pinn_use_huber", True)),
                huber_delta=float(training_cfg.get("pinn_huber_delta", 1.0))
            )
            
            # Get parents dictionary from skeletal structure
            parents_dict = getSkeletalParentsDict()
            
            # Initialize PINN loss module
            self.pinn_loss = PINNLoss(
                parents=parents_dict,
                num_joints=50,
                cfg=pinn_cfg
            )
            
            print(f"PINN Loss initialized with {len(parents_dict)} parent-child pairs")
            print(f"  - lambda_pinn: {self.lambda_pinn}")
            print(f"  - lambda_bone: {pinn_cfg.lambda_bone}")
            print(f"  - lambda_vel: {pinn_cfg.lambda_vel}")
            print(f"  - lambda_acc: {pinn_cfg.lambda_acc}")
            print(f"  - lambda_fk: {pinn_cfg.lambda_fk}")
        else:
            self.pinn_loss = None
            print("PINN Loss is disabled")

    def forward(self, preds, targets, mask: Optional[torch.Tensor] = None):
        """
        Compute loss with optional PINN regularization
        
        Args:
            preds: predicted skeleton (B, T, 150)
            targets: target skeleton (B, T, 150)
            mask: optional mask (B, T) for valid frames
        """
        # Create loss mask from target padding
        loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        # Compute bone length and direction features
        preds_masked_length, preds_masked_direct = get_length_direct(preds_masked)
        targets_masked_length, targets_masked_direct = get_length_direct(targets_masked)

        preds_masked_length = preds_masked_length * loss_mask[:, :, :50]
        targets_masked_length = targets_masked_length * loss_mask[:, :, :50]
        preds_masked_direct = preds_masked_direct * loss_mask[:, :, :150]
        targets_masked_direct = targets_masked_direct * loss_mask[:, :, :150]

        # Calculate base reconstruction loss
        recon_loss = self.criterion(preds_masked, targets_masked) + \
                     0.1 * self.criterion_bone(preds_masked_direct, targets_masked_direct)

        # Add PINN loss if enabled
        if self.use_pinn and self.pinn_loss is not None:
            # Create mask for PINN (B, T) - True for valid frames
            if mask is None:
                # Infer mask from targets: a frame is valid if it's not all padding
                # Check if any coordinate in the frame is non-pad
                pinn_mask = (targets[:, :, 0] != self.target_pad)  # (B, T)
            else:
                pinn_mask = mask
            
            # Compute PINN losses on predictions
            pinn_out = self.pinn_loss(preds, pinn_mask)
            
            # Total PINN loss
            pinn_total = pinn_out["total"]
            
            # Combined loss
            loss = recon_loss + self.lambda_pinn * pinn_total
        else:
            loss = recon_loss

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss


def get_length_direct(trg):
    """
    Compute bone lengths and unit directions from skeleton coordinates
    
    Args:
        trg: skeleton tensor (B, T, 150) - 50 joints x 3 coordinates
    
    Returns:
        lengths: (B, T, num_bones) - length of each bone
        directs: (B, T, 3*num_bones) - unit direction vectors (3D) for each bone
    """
    trg_reshaped = trg.view(trg.shape[0], trg.shape[1], 50, 3)
    trg_list = trg_reshaped.split(1, dim=2)
    trg_list_squeeze = [t.squeeze(dim=2) for t in trg_list]
    skeletons = getSkeletalModelStructure()

    length = []
    direct = []
    for skeleton in skeletons:
        Skeleton_length = torch.norm(trg_list_squeeze[skeleton[0]]-trg_list_squeeze[skeleton[1]], p=2, dim=2, keepdim=True)
        result_length = Skeleton_length
        result_direct = (trg_list_squeeze[skeleton[0]]-trg_list_squeeze[skeleton[1]]) / (Skeleton_length+torch.finfo(Skeleton_length.dtype).tiny)
        direct.append(result_direct)
        length.append(result_length)
    lengths = torch.stack(length, dim=-1).squeeze()
    directs = torch.stack(direct, dim=2).view(trg.shape[0], trg.shape[1], -1)

    return lengths, directs
