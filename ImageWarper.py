class ImageWarper():

    def __init__(self):
        pass
    def get_perspective_transform(self, image, region_of_interest: list, transformed_dimensions: list):
        # Need to switch first and second index, since getPerspectiveTransform method requires this order
        region_of_interest[0], region_of_interest[1] = region_of_interest[1], region_of_interest[0]
        # Make array float32
        region_of_interest = np.array(region_of_interest, np.float32)

        width = transformed_dimensions[0]
        height = transformed_dimensions[1]
        
        transformed_image = np.array([[0,0],[width,0],[0,height],[width,height]], np.float32)

        # Creates transformation matirx to transform region_of_interest dimensions to transformed_image
        M = cv2.getPerspectiveTransform(region_of_interest, transformed_image)
        invM = cv2.getPerspectiveTransform(region_of_interest, transformed_image)

        return (M, invM)

