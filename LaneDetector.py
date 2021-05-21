import cv2
import matplotlib.pyplot as plot
import numpy as np

from ImageProcessing import ImageReader
from ImageProcessing import ImageMasker
from ImageProcessing import ImageWarper
from ImageProcessing import ImageThresholder

from DeepLearning import Model

class LaneDetector:
    

    def __init__(self, image_path: str):
        self.image_reader = ImageReader.ImageReader(image_path)
        # Read image
        self.image = self.image_reader.read_image()



        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.image_dimensions = [self.width, self.height]

        print("Image Dimensions: ", self.image_dimensions)
         
        self.region_of_interest = [[int(0.46*self.width),int(0.54*self.height)],[int(0.52*self.width),int(0.54*self.height)],[int(0.80*self.width),int(1*self.height)], [int(0.13*self.width),int(1*self.height)]]
        self.image_masker = ImageMasker.ImageMasker(self.region_of_interest, self.image)


        self.image_warper = ImageWarper.ImageWarper(self.image_dimensions)

        self.image_thresholder = ImageThresholder.ImageThresholder()

        self.cnn_model = Model.Model()

    
    def plot_gray_image(self, image):
        plot.imshow(image,cmap="gray")
        plot.show()

    def get_masked_image(self):
        masked_image = self.image_masker.apply_mask_on_region_of_interest(self.image)
        return masked_image

    def get_warped_image(self):
        masked_image = self.get_masked_image()
        perspective_matrices = self.image_warper.get_perspective_transform(self.image, self.region_of_interest, self.image_dimensions)
        warped_image = self.image_warper.warp_image(masked_image)
        self.plot_gray_image(warped_image)
        return warped_image

    def get_thresholded_image(self):
        warped_image = self.get_warped_image()
        thresholded_image = self.image_thresholder.adaptive_threshold(warped_image, 255, 1001, -25)
        self.plot_gray_image(thresholded_image)
        return thresholded_image


    def load_training_data(self):
        pass


    def load_labeled_data(self):
        pass

    def train_model(self, model, epochs, batch, input_shape):
        pass




def main():
    lane_detector = LaneDetector("Images/00000456.png")
    lane_detector.get_thresholded_image()
    

if __name__ == "__main__":
    main()


    