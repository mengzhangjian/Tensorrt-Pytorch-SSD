import cv2
import json
import base64
import requests
import os
import numpy as np
from sort import *
from imutils import paths
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


poly = [(550, 105), (1585, 85), (1619, 979),(450, 972)]
line_1 = [(550, 105), (450, 972)]
polygon = Polygon(poly)
pts = np.array(poly)
pts = pts.reshape((-1, 1, 2))

fontC = ImageFont.truetype("./Font/platech.ttf", 60, 0) 


name_reverse= {"pig": 0}
####Tracker Initialization
tracker = Sort()

previous_tracker_id = []

def post_result(cv_img):
    """

    """
    imencoded = cv2.imencode('.jpg', cv_img)[1].tostring()
    data = {"image": base64.b64encode(imencoded).decode('utf-8'), "threshold": 0.6}
    r = requests.post('http://0.0.0.0:8080/pig', data=json.dumps(data), timeout=50)
    result = json.loads(r.text)
    boxes = []
    track_boxes = []
    id = -1
    for r in result["results"]:
        left = r["location"]["left"]
        top = r["location"]["top"]
        width = r["location"]["width"]
        height = r["location"]["height"]
        center_x = left + width // 2
        name = r["name"]
        score = r["score"]
        boxes.append([left, top, left + width, top + height, float(score), name_reverse["pig"]])
        if center_x < 1600 and center_x > 500:
            track_boxes.append([left, top, left + width, top + height, float(score), name_reverse["pig"]])
    
    return boxes, track_boxes

def show_bounding_boxes(img, result):
    """
    """
    for item in result:
        left, top, right, bottom, socre, _ = item
        center_x = left + (right - left) // 2
        center_y = top + (bottom - top) // 2
        point1 = Point(center_x, center_y)
        if polygon.contains(point1):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        else:
             cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])



if __name__ == "__main__":

    cap = cv2.VideoCapture('2.mp4')
    ret = True
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("img", 960, 540)
    counter = 0
    count = 0
    while ret:
        dets = []
        ret, frame = cap.read()
        boxes, dets = post_result(frame)
        show_bounding_boxes(frame, boxes)
        cv2.polylines(frame, [pts], True, [0, 255, 0], 3)
        cv2.line(frame, (550, 105), (450, 972), (0, 0, 255), 3)

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)

        if len(tracks) > 0:
            for box in tracks:
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
                track_id = box[4]
                p0, p1 = (x, y), (x + w, y)
                
                if intersect(p0, p1, line_1[0], line_1[1]):
                    if track_id not in previous_tracker_id:
                        counter += 1
                        previous_tracker_id.append(track_id)
        image = Image.fromarray(frame) 
        draw = ImageDraw.Draw(image)
        draw.text((40, 200),("出栏数: "), (255, 255, 255), font = fontC)
        draw.text((250, 200),(str(counter)), (0, 0, 255), font = fontC)
        # cv2.putText(frame, str(counter), (500, 500), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)
        frame = np.array(image)
        cv2.imshow("Img", frame)
        # img_name = os.path.join("save", str(count) + ".jpg")
        # count += 1
        # cv2.imwrite(img_name, frame)
        cv2.waitKey(1)



