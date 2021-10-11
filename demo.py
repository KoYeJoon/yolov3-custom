from IPython.display import display 
from PIL import Image
from yolo import YOLO
import cv2

def objectDetection(file, model_path, class_path):
    yolo = YOLO(model_path=model_path, classes_path=class_path, anchors_path='model_data/yolo_tiny_anchors.txt')
    model = yolo.get_model()
    model.summary()
    # 이미지 로딩
    image = Image.open(file)
    # 실행
    result_image = yolo.detect_image(image)
    # 실행 결과 표시
    # display(result_image)
    # 안보이면, 
    cv2.imshow('result',result_image)
    cv2.waitKey(0)

def saved_model():
    yolo = YOLO(model_path='model_data/hand/trained_tiny_weights_final.h5', classes_path='data/light/classes.txt', anchors_path = 'model_data/yolo_tiny_anchors.txt')
    model = yolo.get_model()
    model.save('model_data/hand/tinyHandModel.h5')
    
objectDetection('dog2.jpeg', 'model_data/yolo_tiny.h5', 'model_data/coco_classes.txt')