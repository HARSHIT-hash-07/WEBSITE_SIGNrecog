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

def render_skeleton_to_video(skeletons, output_path: str, fps: int = 25):
    """
    Renders a list of 3D skeletons to an mp4 video.
    skeletons: A list/array of shape (frames, 50, 3)
    Output is saved to output_path.
    """
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (650, 650), True)

    for i, skel in enumerate(skeletons):
        frame = np.ones((650, 650, 3), np.uint8) * 255
        
        # Take X, Y coordinates
        joints_2d = np.array(skel)[:, :2]
        
        # Usually from plot_videos.py: it had `* 3` applied if data was divided by 3
        # Assuming our model outputs the normalized data directly without dividing by 3
        # If it looks too small we can multiply by 3 later
        joints_2d = joints_2d * 3
        
        draw_frame_2D(frame, joints_2d)
        
        cv2.putText(frame, "Generated Sign: Live AI Bridge", (150, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        out.write(frame)

    out.release()
    
    # Convert to web-compatible h264 using FFmpeg
    tmp_path = output_path.replace(".mp4", "_tmp.mp4")
    os.rename(output_path, tmp_path)
    os.system(f"ffmpeg -y -i {tmp_path} -vcodec libx264 -pix_fmt yuv420p -preset fast -crf 22 {output_path} -loglevel quiet")
    os.remove(tmp_path)
    
    return output_path
