 python train_yolo.py \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --data /Users/michaelwilliams/PycharmProjects/RAGChat/dataset/dataset.yaml \
  --cfg /Users/michaelwilliams/PycharmProjects/RAGChat/yolov7/cfg/training/yolov7-tiny-custom.yaml \
  --weights '' \
  --name yolov7-tiny-custom \
  --hyp yolov7/data/hyp.scratch.p5.yaml
