import argparse
import sys
import os
import ffmpeg
import pathlib
from rembg.bg import remove
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('best-12180 elaborate kudzu.pt')

#Parse args
parser = argparse.ArgumentParser(description='Applies rembg to the frames of a video')
parser.add_argument('input', type=str, help='Input video')
parser.add_argument('-a', action="store_true", help="Turns on alpha matting during background removal")
parser.add_argument('-af', type=int, default=240, help="Alpha matting foreground threshold")
parser.add_argument('-ab', type=int, default=10, help="Alpha matting background threshold")
parser.add_argument('-ae', type=int, default=10, help="Alpha matting erode size")
parser.add_argument('--skip-extract', action="store_true", help='Skips ffmpeg frame extraction')
parser.add_argument('--skip-process', action="store_true", help='Skips rembg frame processing')
args = parser.parse_args()

#Extract video info
probe = ffmpeg.probe(args.input)
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
width = int(video_stream['width'])
height = int(video_stream['height'])
whstr = str(width) + 'x' + str(height)
framerate = video_stream['avg_frame_rate']

workspace = str(pathlib.Path(__file__).parent.absolute())
files_dir = str(pathlib.Path(__file__).parent.absolute()) + "/" + "frames"
processed_dir = str(pathlib.Path(__file__).parent.absolute()) + "/" + "processed"
inference_dir = str(pathlib.Path(__file__).parent.absolute()) + "/" + "segmented"

#Extract input video frames
if not args.skip_extract:
  if not os.path.isdir(files_dir):
    os.mkdir(files_dir)

  stream = ffmpeg.input(args.input)
  stream = ffmpeg.output(stream, "frames/%04d.png")
  ffmpeg.run(stream)

#Process frames with rembg
if not args.skip_process:
  if not os.path.isdir(processed_dir):
    os.mkdir(processed_dir)
  

  for file in os.listdir(files_dir):
    with open(os.path.join(files_dir, file), "rb") as i:
      with open(os.path.join(processed_dir, file), "wb") as o:
          input = i.read()
          output = remove(input, alpha_matting=args.a, alpha_matting_foreground_threshold=args.af, alpha_matting_background_threshold=args.ab, alpha_matting_erode_size=args.ae)
          o.write(output)
  
model.predict(processed_dir, 
                               save_frames=True,
                  device='mps', imgsz=2048, 
                  conf=0.1, show=False, 
                  save=True, save_txt=True,
                show_boxes=True,
                                    
                classes=[0,2,3,4,5],
                
                project=inference_dir, name='segmented')

#Output video
stream = ffmpeg.input("segmented/%04d.png", r=framerate, f='image2', s=whstr, pix_fmt='yuv420p')
stream = ffmpeg.output(stream, workspace + '/' + args.input + "-bg-removed-segmented.mp4", vcodec='libx264', crf=25)
ffmpeg.run(stream)
