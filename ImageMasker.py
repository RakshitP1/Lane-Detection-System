class ImageMasker:
    
    def __init__(self, region_of_interest):
        self.region_of_interest = region_of_interest

    def set_region_of_interest(self, region_of_interest: list):
        self.region_of_interest = region_of_interest

    def get_region_of_interest(self):
        return self.region_of_interest


    