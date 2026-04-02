import torch
import os
import collections

# Paths
input_path = "/Users/harshit/Documents/WEBSITE_EXPLO/hf_deploy/weights/best.ckpt"
output_path = "/Users/harshit/Documents/WEBSITE_EXPLO/hf_deploy/weights/best_inference.ckpt"

def shrink():
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
    
    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    
    print(f"Original keys: {list(checkpoint.keys())}")
    
    # We only need the model_state for inference!
    if "model_state" in checkpoint:
        print("✅ Found 'model_state'. Extracting...")
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        print("✅ Found 'state_dict'. Extracting...")
        state_dict = checkpoint["state_dict"]
    else:
        print("❌ Could not find model weights. Checking top-level...")
        state_dict = checkpoint

    # Convert all tensors to float32 to be safe and save space if any were float64
    for key in state_dict:
        if isinstance(state_dict[key], torch.Tensor):
            if state_dict[key].dtype == torch.float64:
                state_dict[key] = state_dict[key].to(torch.float32)

    # Re-wrap in a format that your loaders expect
    # Your current loader in sign_bridge_inference.py expects a flat state_dict? 
    # Or does it expect a nested one?
    # Let's check sign_bridge_inference.py line 55:
    # checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
    # Then build_model(..., checkpoint=checkpoint)
    
    # Let's see how build_model uses it.
    # We'll save the whole thing with just model_state.
    inference_checkpoint = {
        "model_state": state_dict,
        "steps": checkpoint.get("steps", 0),
        "total_tokens": checkpoint.get("total_tokens", 0)
    }
    
    print(f"Saving stripped weights to: {output_path}")
    torch.save(inference_checkpoint, output_path)
    
    old_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original Size: {old_size:.2f} MB")
    print(f"New Stripped Size: {new_size:.2f} MB")
    
    if new_size < 1000:
        print("✅ SUCCESS: Model is now under 1GB!")
    else:
        print("⚠️ Warning: Model is still over 1GB.")

if __name__ == "__main__":
    shrink()
