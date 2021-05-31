import cv2

class ImageReader:



    def __init__(self, image_path=None):
        self.image_path = image_path

    def read_image_with_mode(self, mode):
        return cv2.imread(self.image_path, mode)

    def read_image(self):
        return cv2.imread(self.image_path)

    def set_image_path(self, image_path):
        self.image_path = image_path

    