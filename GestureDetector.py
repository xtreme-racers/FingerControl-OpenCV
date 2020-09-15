import numpy as np
import cv2


class FingerControl:
    def __init__(self, x0, y0, x1, y1, movements):
        self.webcam = cv2.VideoCapture(0)
        # Left hand region
        self.region = None
        # Skin color histogram
        self.skincolor_hist = None

        self.marker = []
        self.rect = 9
        self.initRect_x = None
        self.initRect_y = None
        self.initRect_x1 = None
        self.initRect_y1 = None

        # ROI
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        #movement
        self.movements = movements

    # Draw zone for capture player skin tone
    def initRect(self, frame):
        rows, cols, _ = self.region.shape

        self.initRect_x = np.array(
            [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
             12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

        self.initRect_y = np.array(
            [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20,
             9 * cols / 20,
             10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

        self.initRect_x1 = self.initRect_x + 10
        self.initRect_y1 = self.initRect_y + 10

        for i in range(self.rect):
            cv2.rectangle(self.region, (self.initRect_y[i], self.initRect_x[i]),
                          (self.initRect_y1[i], self.initRect_x1[i]),
                          (0, 255, 0), 1)
        return frame

    # Calculate skin tone histogram
    def skinColor(self):
        hsv_frame = cv2.cvtColor(self.region, cv2.COLOR_BGR2HSV)
        roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

        for i in range(self.rect):
            roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[self.initRect_x[i]:self.initRect_x[i] + 10, self.initRect_y[i]:self.initRect_y[i] + 10]

        hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

    # Mask player hand by using the generated skin tone histogram
    def skinColorMasking(self):
        hsv = cv2.cvtColor(self.region, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([hsv], [0, 1], self.skincolor_hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.merge((thresh, thresh, thresh))

        return cv2.bitwise_and(self.region, thresh)

    def contours(self, hist_mask_image):
        gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
        cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cont

    # Center point of the player hand
    def centerPoint(self, max_contour):
        moment = cv2.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return None

    # Get farthest point from the center point
    def farthest_point(self, defects, contour, centroid):
        if defects is not None and centroid is not None:
            s = defects[:, 0][:, 0]
            cx, cy = centroid
            x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
            y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

            xp = cv2.pow(cv2.subtract(x, cx), 2)
            yp = cv2.pow(cv2.subtract(y, cy), 2)
            dist = cv2.sqrt(cv2.add(xp, yp))

            dist_max_i = np.argmax(dist)

            if dist_max_i < len(s):
                farthest_defect = s[dist_max_i]
                farthest_point = tuple(contour[farthest_defect][0])
                #print("x:", np.argmax(cv2.subtract(x, cx)), " y:", np.argmax(cv2.subtract(y, cy)))

                return farthest_point
            else:
                return None

    def process(self):
        hist_mask_image = self.skinColorMasking()

        hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)
        hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=2)

        contour_list = self.contours(hist_mask_image)
        try:
            max_cont = max(contour_list, key=cv2.contourArea)
        except Exception as e:
            #print(e)
            max_cont = None

        cnt_centroid = self.centerPoint(max_cont)
        cv2.circle(self.region, cnt_centroid, 5, [255, 0, 255], -1)

        if max_cont is not None:
            try:
                hull = cv2.convexHull(max_cont, returnPoints=False)
                defects = cv2.convexityDefects(max_cont, hull)
                far_point = self.farthest_point(defects, max_cont, cnt_centroid)
                cv2.circle(self.region, far_point, 5, [0, 0, 255], -1)
                des = (far_point[0]-cnt_centroid[0],far_point[1]-cnt_centroid[1])
            except Exception as e:
                #print(e)
                des = (0,0)
                far_point = (0,0)

            if(des[0]<0 and des[1]<0):
                print(self.movements["left"])
            elif(des[0]>=0 and des[1]<0):
                print(self.movements["up"])
            elif(des[0]>=0 and des[1]>=0):
                print(self.movements["right"])
            else:
                print(self.movements["down"])

            if len(self.marker) < 2:
                self.marker.append(far_point)
            else:
                self.marker.pop(0)
                self.marker.append(far_point)

            self.draw()

        else:
            print("Hand not detected!")

    # Draw debug points
    def draw(self):
        if self.marker is not None:
            for i in range(len(self.marker)):
                cv2.circle(self.region, self.marker[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)

    def start(self):
        skintonemap = False

        while self.webcam.isOpened():
            pressed_key = cv2.waitKey(1)
            _, frame = self.webcam.read()
            frame = cv2.flip(frame, 1)

            self.region = frame[self.x0:self.y0, self.x1:self.y1]
            cv2.rectangle(frame, (self.x0, self.x1), (self.y0, self.y1), (0, 255, 0), 0)

            if pressed_key & 0xFF == ord('z'):
                skintonemap = True
                self.skincolor_hist = self.skinColor()

            if skintonemap:
                #print("good")
                self.process()

            else:
                frame = self.initRect(frame)

            cv2.imshow("GestureControl", frame)

            if pressed_key == 27:
                break

        cv2.destroyAllWindows()
        self.webcam.release()


# if __name__ == "__main__":
#     app = FingerControl(300, 900, 300, 900)
#     app.start()