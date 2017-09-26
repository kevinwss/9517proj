import cv2
import dlib
import numpy

import sys
import os


def get_landmarks(im):
    PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])



def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
    

def get_face_mask(im, landmarks,size):
    FEATHER_AMOUNT = 11
    
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_BROW_POINTS = list(range(17, 22))
    NOSE_POINTS = list(range(27, 35))
    MOUTH_POINTS = list(range(48, 61))
    OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

    mask = numpy.zeros(im.shape[:2], dtype=numpy.float64)
    
    draw_convex_hull(mask,landmarks,color=1)
    xs=[]
    ys=[]

    for ind in range(68):
        xs.append(landmarks[ind,0])
        ys.append(landmarks[ind,1])

    xmin,xmax,ymin,ymax = min(xs),max(xs),min(ys),max(ys)

    mask =mask[ymin:ymax,xmin:xmax]
    im =im[ymin:ymax,xmin:xmax,:]
    
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if(mask[y][x]!=1):
                im[y][x][0] = 255
                im[y][x][1] = 255
                im[y][x][2] = 255

    im = cv2.resize(im,(size[0],size[1]))
    
    return im

def read_im_and_landmarks(fname):
    SCALE_FACTOR = 1
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def main():
    
    size = [300,300]
    
    if (len(sys.argv)==3):
        input_dir = "./input/"
        output_dir = "./output/"
        size[0] = int(sys.argv[1])
        size[1] = int(sys.argv[2])
    elif (len(sys.argv)==5):
        size[0] = int(sys.argv[1])
        size[1] = int(sys.argv[2])
        input_dir = sys.argv[3]
        output_dir = sys.argv[4]
    else:
        print("arg: -input folder -output folder")
        sys.exit()

    
    for img_name in os.listdir(input_dir):
        print(img_name)
        im, landmarks = read_im_and_landmarks(input_dir+img_name)
        mask = get_face_mask (im,landmarks,size)
        print("done")
        cv2.imwrite(output_dir+img_name, mask)
    print("All done")
    

main()
    
    
