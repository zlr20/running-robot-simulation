import cv2 as cv
import numpy as np

def detect(readimg):

    def preprocess(img,channel=1):
        imghsv  = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:,:,channel]
        _,imgbin = cv.threshold(imghsv,127,255,cv.THRESH_BINARY)
        imgbin = cv.GaussianBlur(imgbin,(5,5),0)
        imgedge = cv.Canny(imgbin, 100, 150)
        try:
            circles = cv.HoughCircles(imgedge, cv.HOUGH_GRADIENT, 1, int(min(imgedge.shape)/3),
                                    param1=50, param2=20, minRadius=10, maxRadius=int(max(imgedge.shape)/2))
            circles = np.uint16(np.around(circles))
            return circles[0,:]
        except:
            return []

    def postprocess(img,circles,channel=1):
        h,w,c = img.shape
        imghsv  = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:,:,channel]
        imgbin = cv.adaptiveThreshold(imghsv,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,3,5)
        circles_with_certainty = []
        for i,circle in enumerate(circles):
            canvas = np.zeros((h,w),dtype=np.uint8)
            cv.circle(canvas, (circle[0], circle[1]), circle[2], (255, 255, 255), 2) # (x,y,r)
            overlap = cv.bitwise_and(imgbin, canvas)
            certainty = np.sum(overlap)/(255*2*np.pi*circle[2])
            circles_with_certainty.append({'circle':[circle[0], circle[1],circle[2]],'certainty':certainty})
        
        return circles_with_certainty

    h,w,c = readimg.shape
    h0 = 480
    ratio = h0/h
    img = cv.resize(readimg,(int(w*ratio),h0))
    img = cv.pyrMeanShiftFiltering(img, 10, 50)
    result = []
    for channel in [1,2]:
        circles = preprocess(img,channel=channel)
        circles_with_certainty = postprocess(img,circles,channel=channel)
        result.extend(circles_with_certainty)
    result = sorted(result, key=lambda r: r['certainty'])
    circle = result[-1]['circle']
    return int(circle[2]/ratio),int(circle[0]/ratio),int(circle[1]/ratio)
    