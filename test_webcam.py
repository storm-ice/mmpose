import cv2

# 创建VideoCapture对象并打开摄像头（通常摄像头编号为0）
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头捕获的帧
    ret, frame = cap.read()

    # 显示摄像头画面在屏幕上
    cv2.imshow('Camera Feed', frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()