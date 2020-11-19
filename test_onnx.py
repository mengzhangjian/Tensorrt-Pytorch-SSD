import onnxruntime
import numpy as np
import cv2
import sys
# 1. Make session
session = onnxruntime.InferenceSession('ssd-mobilenet.onnx')


def main(frame):
    # Load image
    width, height = 300, 300
    frame = cv2.resize(frame, (300, 300))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255

    # HWC to NHWC
    image_data = np.expand_dims(img.transpose((2, 0, 1)), 0).astype(np.float32)

    # 2. Get input/output name
    input_name = session.get_inputs()[0].name   # 'image'
    output_name_scores = session.get_outputs()[0].name     # 'scores'
    output_name_boxes = session.get_outputs()[1].name   # 'boxes'

    # 3. Run
    outputs_index = session.run([], {input_name: image_data})

    # Result
    output_scores = outputs_index[0]
    output_boxes = outputs_index[1]

    # Check input/output name
    print(input_name, output_name_boxes, output_name_scores)
    print(output_scores)
    print(output_scores.size)
    for i, scores in enumerate(output_scores[0]):
        max_index = np.argmax(scores)
        if max_index != 0:
            boxes = output_boxes[0][i]
            print(boxes[0], boxes[1], boxes[2], boxes[3])
            left = boxes[0] * height
            top = boxes[1] * width
            right = boxes[2] * height
            bottom = boxes[3] * width
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
            right = min(width, np.floor(right + 0.5).astype('int32'))
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 4)
    cv2.imshow('annotated', frame)


if __name__ == '__main__':

    img = cv2.imread(sys.argv[1])
    main(img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
