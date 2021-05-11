import cv2

class ImageThresholder:


    def adaptive_threshold(self, image, max_value, block_size, threshold_value):
        thresholded_image = cv2.adaptiveThreshold(image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, threshold_value)
        return thresholded_image