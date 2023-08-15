import locale
import ultralytics
from ultralytics import YOLO


def getpreferredencoding(do_setlocale = True):
    return "UTF-8"

locale.getpreferredencoding = getpreferredencoding
print(ultralytics.checks())
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=400, patience=70, pretrained=True, verbose=True, save_period=50, lr0=1e-2, lrf=1e-3)