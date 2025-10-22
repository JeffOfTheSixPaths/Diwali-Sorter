# %%
!nvidia-smi

# %%
from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()

# %%


from roboflow import Roboflow
rf = Roboflow(api_key="")
project = rf.workspace("thespace-uoy2j").project("diwali-sorter-fzmsb")
version = project.version(3)
dataset = version.download("yolov11")
                

# %%
!yolo task=detect mode=train model=yolo11m.pt data=/home/gohith/Programs/DiwaliYOLOV8/Diwali-Sorter-3/data.yaml epochs=10 imgsz=640

# %%



