
import json
import numpy as np
import cv2
import matplotlib.pyplot as plot



class PolygonMaker:
    

    def __init__(self, json_data=None):
        self.json_data = json_data


    
    #load polygon information from json
    def load_json_data(self, json_path):
        data = None
        with open(json_path) as f:
            data = json.load(f)

            # Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
            # print(data)
        self.json_data = data

    def extract_polygon_points(self):
        all_labeled_polygons = []
        for image_data in self.json_data:
            polygon_data = image_data["Label"]["objects"][0]["polygon"]
            polygon_point_list = []
            for point in polygon_data:
                list_point = np.array([int(point["x"]), int(point["y"])], np.int32)
                polygon_point_list.append(list_point)

            all_labeled_polygons.append(np.array(polygon_point_list, np.int32))

        return all_labeled_polygons


    def draw_labeled_polygons(self, all_labeled_polygons, image_shape=(480,640)):
        image_number = 0
        for labeled_polygon in all_labeled_polygons:
            
            polygon_image = np.zeros(image_shape)
            cv2.fillPoly(polygon_image, pts = [labeled_polygon], color =(255,0,255))
            cv2.imshow("filledPolygon", polygon_image)
            plot.plot()
            cv2.imwrite('../DeepLearning/labeled_images/' + str(image_number) + ".png", polygon_image)
            image_number+=1



polygon_maker = PolygonMaker()
polygon_maker.load_json_data("labeled_image_json.json")
all_labeled_polygons = polygon_maker.extract_polygon_points()
polygon_maker.draw_labeled_polygons(all_labeled_polygons)






        


    

