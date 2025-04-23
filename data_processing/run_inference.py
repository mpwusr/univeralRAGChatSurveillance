# import a utility function for loading Roboflow models
from inference import get_model
# import supervision to visualize our results
import supervision as sv
# import cv2 to helo load our image
import cv2
from inference.models.utils import ModelDependencyMissing
import warnings

# Suppress Roboflow optional model dependency warnings
warnings.filterwarnings("ignore", message="Your `inference` configuration does not support")

# Suppress warnings about optional model dependencies
warnings.filterwarnings("ignore", category=ModelDependencyMissing)

# Suppress warnings about unavailable execution providers
warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider' is not in available provider names.")
warnings.filterwarnings("ignore", message="Specified provider 'OpenVINOExecutionProvider' is not in available provider names.")
# define the image url to use for inference
image_file = "../surveillance/incoming_images/apt1.jpg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_model(model_id="yolov8n-640")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results[0].model_dump(by_alias=True, exclude_none=True))

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)