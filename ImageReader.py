import cv2

class ImageReader:



    def __init__(self, image):
        self.image = image

    def read_image(self, mode):
        return cv2.imread(self.image, mode)