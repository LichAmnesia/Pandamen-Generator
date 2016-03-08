# -*- coding: utf-8 -*-

# --------------------------------------------
# Author: Lich_Amnesia <alwaysxiaop@gmail.com>
# Date: 2016-03-08
# --------------------------------------------



import os
import cv2
from PIL import Image,ImageDraw
# log程序初始化
from logger.mylogger import Logger
log_main = Logger.get_logger(__file__)
# ---------------------------------------------
## 训练文件所在位置文件夹初始化以及文件所在位置
harr_path = os.path.join(__file__.rsplit('\\',1)[0],'haarcascades')
img_path = __file__.rsplit('\\',1)[0]
log_main.info('harr_path = {0}; img_path = {1}'.format(harr_path,img_path))
# ---------------------------------------------

#detectFaces()返回图像中所有人脸的矩形坐标（矩形左上、右下顶点）
#使用haar特征的级联分类器haarcascade_frontalface_default.xml，在haarcascades目录下还有其他的训练好的xml文件可供选择。
#注：haarcascades目录下训练好的分类器必须以灰度图作为输入。
def detectFaces(image_file):
    img = cv2.imread(image_file)
    face_cascade = cv2.CascadeClassifier(os.path.join(harr_path,"haarcascade_frontalface_default.xml"))
    # log_main.info('img.ndim = {0}'.format(img.ndim))
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result


#保存人脸图
def saveFaces(image_file):
    faces = detectFaces(image_file)
    count = 0
    if faces:
        #将人脸保存在save_dir目录下。
        #Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
        save_dir = os.path.join(img_path,'face_output')
        image_name = image_file.rsplit('\\',1)[1]
        log_main.info('face_output = {0}'.format(save_dir))

        for (x1,y1,x2,y2) in faces:
            save_file = os.path.join(save_dir, image_name.split('.')[0] + "_" + str(count) + ".jpg")
            Image.open(image_file).crop((x1,y1,x2,y2)).save(save_file)
            count += 1
    return count


#在原图像上画矩形，框出所有人脸。
#调用Image模块的draw方法，Image.open获取图像句柄，ImageDraw.Draw获取该图像的draw实例，然后调用该draw实例的rectangle方法画矩形(矩形的坐标即
#detectFaces返回的坐标)，outline是矩形线条颜色(B,G,R)。
#注：原始图像如果是灰度图，则去掉outline，因为灰度图没有RGB可言。drawEyes、detectSmiles也一样。
def drawFaces(image_file):
    faces = detectFaces(image_file)
    if faces:
        img = Image.open(image_file)
        draw_instance = ImageDraw.Draw(img)
        for (x1,y1,x2,y2) in faces:
            draw_instance.rectangle((x1,y1,x2,y2), outline=(255, 0,0))
        image_name = image_file.rsplit('\\',1)[1]
        img.save(os.path.join(img_path,'drawfaces_'+image_name))



#检测眼睛，返回坐标
#由于眼睛在人脸上，我们往往是先检测出人脸，再细入地检测眼睛。故detectEyes可在detectFaces基础上来进行，代码中需要注意“相对坐标”。
#当然也可以在整张图片上直接使用分类器,这种方法代码跟detectFaces一样，这里不多说。
def detectEyes(image_file):
    eye_cascade = cv2.CascadeClassifier(os.path.join(harr_path,'haarcascade_eye.xml'))
    faces = detectFaces(image_file)

    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = []
    for (x1,y1,x2,y2) in faces:
        roi_gray = gray[y1:y2, x1:x2]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.3,2)
        for (ex,ey,ew,eh) in eyes:
            # y 表示的是纵向
            if y1 + ey > img.shape[0] / 2:
                continue
            if ew < img.shape[0] / 5:
                continue
            result.append((x1+ex,y1+ey,x1+ex+ew,y1+ey+eh))
    return result


#在原图像上框出眼睛.
def drawEyes(image_file):
    eyes = detectEyes(image_file)
    log_main.info(eyes)
    if eyes:
        img = Image.open(image_file)
        draw_instance = ImageDraw.Draw(img)
        for (x1,y1,x2,y2) in eyes:
            draw_instance.rectangle((x1,y1,x2,y2), outline=(0, 0,255))
        image_name = image_file.rsplit('\\',1)[1]
        log_main.info('image_name = {0}'.format(image_name))
        img.save(os.path.join(img_path,'draweyes_'+image_name))


#检测鼻子，返回坐标
#由于眼睛在人脸上，我们往往是先检测出人脸，再细入地检测眼睛。故detectEyes可在detectFaces基础上来进行，代码中需要注意“相对坐标”。
#当然也可以在整张图片上直接使用分类器,这种方法代码跟detectFaces一样，这里不多说。
def detectNose(image_file):
    nose_cascade = cv2.CascadeClassifier(os.path.join(harr_path,'haarcascade_mcs_nose.xml'))
    faces = detectFaces(image_file)

    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = []
    for (x1,y1,x2,y2) in faces:
        roi_gray = gray[y1:y2, x1:x2]
        noses = nose_cascade.detectMultiScale(roi_gray,1.3,2)
        for (ex,ey,ew,eh) in noses:
            result.append((x1+ex,y1+ey,x1+ex+ew,y1+ey+eh))
    return result

def drawNose(image_file):
    noses = detectNose(image_file)
    if noses:
        img = Image.open(image_file)
        draw_instance = ImageDraw.Draw(img)
        for (x1,y1,x2,y2) in noses:
            draw_instance.rectangle((x1,y1,x2,y2), outline=(0, 0,255))
        image_name = image_file.rsplit('\\',1)[1]
        log_main.info('drawNose_ = {0}'.format(image_name))
        img.save(os.path.join(img_path,'drawNose_'+image_name))

#检测眼睛，返回坐标
#由于眼睛在人脸上，我们往往是先检测出人脸，再细入地检测眼睛。故detectEyes可在detectFaces基础上来进行，代码中需要注意“相对坐标”。
#当然也可以在整张图片上直接使用分类器,这种方法代码跟detectFaces一样，这里不多说。
def detectMouth(image_file):
    mouth_cascade = cv2.CascadeClassifier(os.path.join(harr_path,'haarcascade_mcs_mouth.xml'))
    faces = detectFaces(image_file)

    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = []
    for (x1,y1,x2,y2) in faces:
        roi_gray = gray[y1:y2, x1:x2]
        mouths = mouth_cascade.detectMultiScale(roi_gray,1.3,2)
        for (ex,ey,ew,eh) in mouths:
            if y1 + ey < img.shape[0] / 2:
                continue
            if ew < img.shape[0] / 6:
                continue
            result.append((x1+ex,y1+ey,x1+ex+ew,y1+ey+eh))
    return result

def drawMouth(image_file):
    mouths = detectMouth(image_file)
    if mouths:
        img = Image.open(image_file)
        draw_instance = ImageDraw.Draw(img)
        for (x1,y1,x2,y2) in mouths:
            draw_instance.rectangle((x1,y1,x2,y2), outline=(0, 0,255))
        image_name = image_file.rsplit('\\',1)[1]
        log_main.info('drawMouth_ = {0}'.format(image_name))
        img.save(os.path.join(img_path,'drawMouth_'+image_name))

def faceFilter(face_file):
    log_main.info(face_file)
    img = cv2.imread(face_file)


    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
    # 先做中值滤波，canny找边缘，然后去边缘
    median_result = cv2.medianBlur(gray,7)
    canny_result = cv2.Canny(median_result,125,180);
    # log_main.info('{0},{1}'.format(median_result.shape,canny_result.shape))
    result = cv2.adaptiveThreshold(median_result,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    img_shape = result.shape
    ## 检测眼睛
    eyes = detectEyes(face_file)
    eysemaxy = img_shape[0] / 2
    if eyes != None:
        minx = 10000
        maxx = -1000
        miny = 10000
        maxy = -1000
        for bx,by,ex,ey in eyes:
            minx = min(bx,minx)
            maxx = max(ex,maxx)
            miny = min(by,miny)
            maxy = max(ey,maxy)
        eysemaxy = maxy
        log_main.info(eyes)
        log_main.info('{0} {1} {2}'.format(minx,miny,maxy))
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if i <= miny or j <= minx or j >= maxx:
                    result[i][j] = 255

    mouths = detectMouth(face_file)
    if mouths != None:
        minx = 10000
        maxx = -1000
        miny = 10000
        maxy = -1000
        for bx,by,ex,ey in mouths:
            minx = min(bx,minx)
            maxx = max(ex,maxx)
            miny = min(by,miny)
            maxy = max(ey,maxy)
        log_main.info(mouths)
        log_main.info('{0} {1} {2}'.format(minx,miny,maxy))
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if  (i >= eysemaxy and j <= minx) or (i >= eysemaxy and j >= maxx) or (i >= maxy):
                    # continue
                    result[i][j] = 255
    if mouths == [] and eyes == []:
        log_main.error("eyes and mouhs detected error!!")
    else:
        log_main.info('{0},{1}'.format(mouths,eyes))

    # 做第二遍中值滤波和边缘检测，并且把在边缘的部分给消掉
    result = cv2.medianBlur(result,3)
    canny_result = cv2.Canny(result,30,100);
    pil_im = Image.fromarray(result)
    # pil_im.show()
    face_name = face_file.rsplit('\\',1)[1]
    face_file = os.path.join(os.path.join(img_path,'face_output'),'output_'+face_name)
    pil_im.save(face_file)

    return face_file

# 合成图片img是输出人脸图片， base_file是要合成的图片
def merge(face_file, base_file):
    base_img = cv2.imread(base_file)
    face_img = cv2.imread(face_file)
    log_main.info('{0},{1}'.format(face_img.shape,base_img.shape))
    face_img = face_img[face_img.shape[0]/7:face_img.shape[0]/7*6,face_img.shape[1]/7:face_img.shape[1]/7*6]
    face_img = cv2.resize(face_img,(200,210))
    base_img[70:280,100:300] = face_img

    pil_img = Image.fromarray(base_img)
    face_name = face_file.rsplit('\\',1)[1]
    face_file = os.path.join(os.path.join(img_path,'face_output'),'merge_'+face_name)
    pil_img.save(face_file)
    log_main.info('{0},{1}'.format(face_img.shape,base_img.shape))

if __name__ == '__main__':
    img_name = 'meizi.jpg'
    img_file = os.path.join(__file__.rsplit('\\',1)[0],img_name)
    log_main.info('{0}'.format(img_file))
    face_count = saveFaces(img_file)

    base_file = os.path.join(__file__.rsplit('\\',1)[0],'base.jpg')
    if face_count == 0:
        log_main.info('face_count = {0}, no face detected!'.format(face_count))
    else :
        log_main.info('face_count = {0},{1} face detected!'.format(face_count,face_count))
        for count in range(face_count):
            face_file = os.path.join(os.path.join(__file__.rsplit('\\',1)[0],'face_output'),img_name.split('.')[0]+'_' + str(count) + '.jpg')
            log_main.info(face_file)
            # drawSmiles(face_file)
            face_file = faceFilter(face_file)
            merge(face_file, base_file)
    # drawFaces(img_file)
    # drawEyes(img_file)
    # drawSmiles(face_file)

    # log_main.error('sss, {0}'.format(2))

