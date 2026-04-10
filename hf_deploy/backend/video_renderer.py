import os
import cv2
import math
import numpy as np

def draw_line(im, joint1, joint2, c=(0, 0, 255),t=1, width=3):
    thresh = -100
    if joint1[0] > thresh and  joint1[1] > thresh and joint2[0] > thresh and joint2[1] > thresh:
        center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))
        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2))/2)
        angle = math.degrees(math.atan2((joint1[0] - joint2[0]),(joint1[1] - joint2[1])))
        cv2.ellipse(im, center, (width,length), -angle,0.0,360.0, c, -1)

def draw_frame_2D(frame, joints):
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from SignIDD_CodeFiles.helpers import getSkeletalModelStructure
    draw_line(frame, [1, 650], [1, 1], c=(0,0,0), t=1, width=1)
    offset = [350, 250]
    skeleton = np.array(getSkeletalModelStructure())
    number = skeleton.shape[0]

    # Increase the size and position of the joints
    joints_scaled = joints * 10 * 12 * 2
    joints_scaled = joints_scaled + np.ones((50, 2)) * offset

    for j in range(number):
        c = (0, 0, 0) # Black bones format like original script
        draw_line(frame, [joints_scaled[skeleton[j, 0]][0], joints_scaled[skeleton[j, 0]][1]],
                  [joints_scaled[skeleton[j, 1]][0], joints_scaled[skeleton[j, 1]][1]], c=c, t=1, width=1)

def render_skeleton_to_video(skeletons, output_path: str, fps: int = 25, mode: str = "standard"):
    """
    Renders a list of 3D skeletons to an mp4 video with Auto-Scaling.
    skeletons: A list/array of shape (frames, 50, 3)
    """
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (650, 650), True)

    # Convert to numpy for easier manipulation
    skeletons_np = np.array(skeletons) # (F, 50, 3)
    
    # 1. AUTO-SCALING LOGIC
    # Find the bounding box across all frames to keep scaling consistent
    all_joints_2d = skeletons_np[:, :, :2] # (F, 50, 2)
    min_coords = np.min(all_joints_2d, axis=(0, 1))
    max_coords = np.max(all_joints_2d, axis=(0, 1))
    center = (min_coords + max_coords) / 2
    
    # Calculate scale factor to fit 80% of the 650x650 canvas
    range_coords = max_coords - min_coords
    max_range = np.max(range_coords)
    if max_range < 1e-6: max_range = 1.0 # Avoid div by zero
    scale = (650 * 0.7) / max_range

    for i, skel in enumerate(skeletons):
        frame = np.ones((650, 650, 3), np.uint8) * 245 # Slightly off-white
        
        # 2. Apply Auto-Scale and Center
        joints_2d = (np.array(skel)[:, :2] - center) * scale
        joints_2d = joints_2d + np.array([325, 325]) # Move to center of canvas
        
        # Draw the frame (we pass the pre-scaled joints)
        draw_frame_2D_v2(frame, joints_2d)
        
        label = "Generated Sign: SignBridge HQ" if mode == "hq" else "Generated Sign: Live AI Bridge"
        cv2.putText(frame, label, (130, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        out.write(frame)

    out.release()
    
    # Convert to web-compatible h264 using FFmpeg
    tmp_path = output_path.replace(".mp4", "_tmp.mp4")
    if os.path.exists(output_path):
        os.rename(output_path, tmp_path)
        os.system(f"ffmpeg -y -i {tmp_path} -vcodec libx264 -pix_fmt yuv420p -preset fast -crf 22 {output_path} -loglevel quiet")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return output_path

def draw_frame_2D_v2(frame, joints_scaled):
    """Specific version that takes ALREADY scaled joints to avoid double scaling."""
    from SignIDD_CodeFiles.helpers import getSkeletalModelStructure
    skeleton = np.array(getSkeletalModelStructure())
    
    for j in range(skeleton.shape[0]):
        joint1 = joints_scaled[skeleton[j, 0]]
        joint2 = joints_scaled[skeleton[j, 1]]
        draw_line(frame, joint1, joint2, c=(40, 40, 40), t=1, width=2)
