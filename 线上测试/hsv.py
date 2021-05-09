#霍夫圆检测
import cv2 as cv
import numpy as np

def bi_demo(image):   #双边滤波
   dst = cv2.bilateralFilter(image, 0, 100, 15)
   return dst

def shift_demo(image):   #均值迁移
   dst = cv2.pyrMeanShiftFiltering(image, 10, 50)
   return dst


def preprocess(img,channel=1,debug=False):
    imghsv  = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:,:,channel]
    _,imgbin = cv.threshold(imghsv,127,255,cv.THRESH_BINARY)
    imgbin = cv.GaussianBlur(imgbin,(5,5),0)
    imgedge = cv.Canny(imgbin, 100, 150)
    if debug:
        cv.imwrite('channel'+str(channel)+'_1.png',imgbin)
        cv.imwrite('channel'+str(channel)+'_2.png',imgedge)
    try:
        circles = cv.HoughCircles(imgedge, cv.HOUGH_GRADIENT, 1, int(min(imgedge.shape)/3),
                                param1=50, param2=20, minRadius=10, maxRadius=int(max(imgedge.shape)/2))
        circles = np.uint16(np.around(circles))
        return circles[0,:]
    except:
        return []

def postprocess(img,circles,channel=1,debug=False):
    h,w,c = img.shape
    imghsv  = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:,:,channel]
    imgbin = cv.adaptiveThreshold(imghsv,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,3,5)
    if debug:
        cv.imwrite('channel'+str(channel)+'_3.png',imgbin)
    
    circles_with_certainty = []
    for i,circle in enumerate(circles):
        canvas = np.zeros((h,w),dtype=np.uint8)
        cv.circle(canvas, (circle[0], circle[1]), circle[2], (255, 255, 255), 2) # (a,b,r)
        overlap = cv.bitwise_and(imgbin, canvas)
        if debug:
            cv.imwrite('channel'+str(channel)+'_4_'+str(i)+'.png',overlap)
        certainty = np.sum(overlap)/(255*2*np.pi*circle[2])
        circles_with_certainty.append({'circle':[circle[0], circle[1],circle[2]],'certainty':certainty})
    
    return circles_with_certainty
        




if __name__ == '__main__':
    # 读图
    img = cv.imread('./imgs/easy/1.png')
    h,w,c = img.shape
    h0 = 480
    ratio = h0/h
    img = cv.resize(img,(int(w*ratio),h0))
    #img = cv.bilateralFilter(img, 0, 100, 15)
    img = cv.pyrMeanShiftFiltering(img, 10, 50)

    result = []
    for channel in [1,2]:
        circles = preprocess(img,channel=channel,debug=0)
        circles_with_certainty = postprocess(img,circles,channel=channel,debug=0)
        result.extend(circles_with_certainty)

    result = sorted(result, key=lambda r: r['certainty'])

    circle = result[-1]['circle']

    cv.circle(img, (circle[0], circle[1]), circle[2], (0, 0, 255), 2) # (a,b,r)


    cv.imwrite('final.png',img)