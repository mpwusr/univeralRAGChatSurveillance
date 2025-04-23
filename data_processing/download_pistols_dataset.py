from roboflow import Roboflow
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace("wpnv3-p1t0ls").project("wpndts_3js_psvl")
dataset = project.version(1).download("yolov8")  # Replace 'yolov8' with your desired format
