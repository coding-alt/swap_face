import cv2
import glob
import psutil
import uuid
import random
import os
import gradio as gr
import argparse
from pathlib import Path
from opennsfw2 import predict_image as face_check
from core.swapper import process_img, process_video
from core.config import get_face
from core.utils import detect_fps, set_fps, create_video, add_audio, extract_frames


pool = None
args = {}

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='use gpu', dest='gpu', action='store_true', default=True)
parser.add_argument('--keep-frames', help='keep frames directory', dest='keep_frames', action='store_true', default=False)
parser.add_argument('--max-cores', help='number of cores to use', dest='cores_count', type=int, default=max(psutil.cpu_count() - 2, 2))

for name, value in vars(parser.parse_args()).items():
    args[name] = value

sep = "/"
if os.name == "nt":
    sep = "\\"

def swap_face_img(source_img, target_img):
    print(source_img, target_img)

    print("detecting face...")
    test_face = get_face(cv2.imread(source_img))
    if not test_face:
        print("\n[WARNING] No face detected in source image. Please try with another one.\n")
        return
    
    print("detecting face in target image...")
    if face_check(target_img) > 0.7:
        quit("[WARNING] Unable to determine location of the face in the target. Please make sure the target isn't wearing clothes matching to their skin.")
    
    output_img = '/tmp/' + uuid.uuid4().hex + '.png'

    print("swapping face...")
    process_img(source_img, target_img, output_img)
    print("swap successful!")

    return output_img
    
def swap_face_video(source_img, target_path, limit_fps=False):
    video_name_full = target_path.split("/")[-1]
    video_name = os.path.splitext(video_name_full)[0]
    output_dir = os.path.dirname(target_path) + "/" + video_name
    Path(output_dir).mkdir(exist_ok=True)

    print("detecting video's FPS...")
    fps, exact_fps = detect_fps(target_path)

    if limit_fps:
        new_target_video = '/tmp/' + uuid.uuid4().hex + '.mp4'
        set_fps(target_path, new_target_video, 30)
        target_path, exact_fps = new_target_video, 30

    print("extracting frames...")
    extract_frames(target_path, output_dir)
    frame_paths = tuple(sorted(
        glob.glob(output_dir + "/*.png"),
        key=lambda x: int(x.split(sep)[-1].replace(".png", ""))
    ))
    threshold = len(['frame_args']) if len(args['frame_paths']) <= 10 else 10
    for i in range(threshold):
        if face_check(random.choice(frame_paths)) > 0.8:
            quit("[WARNING] Unable to determine location of the face in the target. Please make sure the target isn't wearing clothes matching to their skin.")
    
    print("processing video...")
    if args['gpu']:
        process_video(args['source_img'], frame_paths)
    else:
        n = len(frame_paths)//(args['cores_count'])
        processes = []
        for i in range(0, len(frame_paths), n):
            p = pool.apply_async(process_video, args=(source_img, frame_paths[i:i+n],))
            processes.append(p)
        for p in processes:
            p.get()
        pool.close()
        pool.join()

    output_video = '/tmp/' + uuid.uuid4().hex + '.mp4'

    print("creating video...")
    create_video(video_name, exact_fps, output_dir)
    print("adding audio...")
    add_audio(output_dir, target_path, video_name_full, args['keep_frames'], output_video)
    print("\n\nVideo saved as:", output_video, "\n\n")
    print("swap successful!")

    return output_video

iface_image = gr.Interface(
    fn=swap_face_img,
    inputs=[
        gr.Image(type='filepath', label="人脸图片"),
        gr.Image(type='filepath', label="换脸图片")
    ],
    outputs=gr.Image(label="合成图片"),
    allow_flagging = "never",
)

iface_video = gr.Interface(
    fn=swap_face_video,
    inputs=[
        gr.Image(type='filepath', label="人脸图片"),
        gr.Video(label="换脸视频"),
        gr.inputs.Checkbox(label="帧率限制为30", default=False)
    ],
    outputs='video',
    allow_flagging = "never",
)

app = gr.TabbedInterface(
    interface_list=[iface_image, iface_video],
    tab_names=["图片换脸", "视频换脸"],
    title="AI换脸",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green
    )
).launch(share=False, debug=True, server_name="0.0.0.0")