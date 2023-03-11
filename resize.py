import os
import cv2

# 设置图片路径
img_path = r"E:\python\python_project\mogushujuji\yuantu"
path = os.path.join(img_path)
img_list = os.listdir(path)

index = 0
for i in img_list:
    print(os.path.join(path, i))
    old_img = cv2.imread(os.path.join(path, i))
    new_img = cv2.resize(old_img, (300, 400))
    index = index + 1
    cv2.imwrite(r"E:\python\python_project\mogushujuji\image\{}.jpg".format(index), new_img)
