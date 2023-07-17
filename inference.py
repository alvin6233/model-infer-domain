from PIL import Image
from openxlab.model import inference
import numpy as np
import time

from mmpretrain.apis import ImageClassificationInferencer

# print('初始化延迟2分钟')
# time.sleep(120)
# print('初始化延迟2分钟结束')
try:
    result = inference("meijiawen1/test-image", ["./demo_text_ocr.jpg"])
except Exception as e:
    print(f'error:{e}')
model_name = "resnet50_8xb32_in1k"
image_inferencer = ImageClassificationInferencer(model_name)


def post_process(out):
    out['pred_scores'] = out['pred_scores'].tolist()
    return out


class Inferencer():

    def __call__(self, image: Image.Image):
        outputs = image_inferencer(np.asarray(image))

        return [post_process(out) for out in outputs]


inferencer = Inferencer()
