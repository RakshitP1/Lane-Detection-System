import cv2
import matplotlib.pyplot as plot
import numpy as np

from HelperClasses import ImageReader
from HelperClasses import ImageMasker
from HelperClasses import ImageWarper

class LaneDetector:
    

    def __init__(self, image_path: str):
        self.image_reader = ImageReader.ImageReader(image_path)
        # Read image
        self.image = self.image_reader.read_image()

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.transformed_dimensions = [self.width, self.height]
         
        self.region_of_interest = [[int(0.46*self.width),int(0.54*self.height)],[int(0.52*self.width),int(0.54*self.height)],[int(0.80*self.width),int(1*self.height)], [int(0.13*self.width),int(1*self.height)]]
        self.image_masker = ImageMasker.ImageMasker(self.region_of_interest, self.image)


        self.image_warper = ImageWarper.ImageWarper(self.transformed_dimensions)

    
    def plot_gray_image(self, image):
        plot.imshow(image,cmap="gray")
        plot.show()

    def plot_warped_image(self):
        masked_image = self.image_masker.apply_mask_on_region_of_interest(self.image)
        perspective_matrices = self.image_warper.get_perspective_transform(self.image, self.region_of_interest, self.transformed_dimensions)
        warped_image = self.image_warper.warp_image(self.image)
        self.plot_gray_image(warped_image)
        return warped_image

    





    









def main():
    lane_detector = LaneDetector("Images/00000456.png")
    lane_detector.plot_warped_image()
    

if __name__ == "__main__":
    main()


    