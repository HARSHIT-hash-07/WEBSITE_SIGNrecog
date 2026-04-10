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
    Renders skeleton frames to an MP4, using the SAME scaling as the original
    plot_videos.py: joints * 3 * 240 + offset [350, 250].
    
    skeletons: list of shape (frames, 50, 3)
    """
    import sys as _sys
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    from SignIDD_CodeFiles.helpers import getSkeletalModelStructure

    skeleton = np.array(getSkeletalModelStructure())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (650, 650), True)

    for skel in skeletons:
        frame = np.ones((650, 650, 3), np.uint8) * 255

        # Exact same pipeline as plot_videos.py draw_frame_2D:
        #   joints * 3      (undo the /3 normalization from data prep)
        #   then * 240      (= 10 * 12 * 2, the display scale)
        #   then + [350, 250]  (center offset)
        joints_2d = np.array(skel)[:, :2] * 3  # (50, 2)
        joints_scaled = joints_2d * 240
        joints_offset = joints_scaled + np.array([350, 250])

        for j in range(skeleton.shape[0]):
            j1 = joints_offset[skeleton[j, 0]]
            j2 = joints_offset[skeleton[j, 1]]
            draw_line(frame, j1, j2, c=(0, 0, 0), t=1, width=2)

        label = "SignBridge HQ" if mode == "hq" else "Live AI Bridge"
        cv2.putText(frame, label, (230, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        out.write(frame)

    out.release()

    # Convert to web-compatible H.264
    tmp_path = output_path.replace(".mp4", "_tmp.mp4")
    if os.path.exists(output_path):
        os.rename(output_path, tmp_path)
        os.system(f"ffmpeg -y -i {tmp_path} -vcodec libx264 -pix_fmt yuv420p -preset fast -crf 22 {output_path} -loglevel quiet")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return output_path
