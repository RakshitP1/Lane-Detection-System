import numpy as np
import cv2

class ImageMasker:
    
    def __init__(self, region_of_interest, image):
        self.region_of_interest = region_of_interest
        self.stencil = np.zeros_like(image)

    def set_region_of_interest(self, region_of_interest: list):
        self.region_of_interest = region_of_interest

    def get_region_of_interest(self):
        return self.region_of_interest

    def set_stencil(self, image):
        self.stencil =  np.zeros_like(image)

    def get_stencil(self):
        return self.stencil

    def apply_mask_on_region_of_interest(self, image):
        polygon = np.array(self.region_of_interest)
        cv2.fillConvexPoly(self.stencil, polygon, 1)
        self.image = cv2.bitwise_and(image[:,:,0], image[:,:,0], mask=stencil)
        return self.image
    