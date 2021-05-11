import cv2
import numpy as np

class ImageWarper():

    def __init__(self, transformed_dimensions: list):
        self.tranformation_matrix = None
        self.inverse_tranformation_matrix = None

        self.width = transformed_dimensions[0]
        self.height = transformed_dimensions[1]

        
    def get_perspective_transform(self, image, region_of_interest: list, transformed_dimensions: list):
        # Need to switch first and second index, since getPerspectiveTransform method requires this order
        region_of_interest[0], region_of_interest[1] = region_of_interest[1], region_of_interest[0]
        # Make array float32
        region_of_interest = np.array(region_of_interest, np.float32)

        
        
        transformed_image = np.array([[0,0],[self.width,0],[0,self.height],[self.width,self.height]], np.float32)

        # Creates transformation matirx to transform region_of_interest dimensions to transformed_image
        self.tranformation_matrix = cv2.getPerspectiveTransform(region_of_interest, transformed_image)
        self.inverse_transformation_matrix = cv2.getPerspectiveTransform(region_of_interest, transformed_image)

        return (tranformation_matrix, inverse_transformation_matrix)

    def warp_image(self, image):

        warped_image = cv2.warpPerspective(image, self.tranformation_matrix, (self.width,self.height), flags=(cv2.INTER_LINEAR))
        return warped_image



    

