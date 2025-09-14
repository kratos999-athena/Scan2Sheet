# referenced this portion , kept it as it as 
class LineExtractor:
    def __init__(self, inverted_image):
        self.inverted_image = inverted_image
        self.vertical_lines_eroded_image = None
        self.horizontal_lines_eroded_image= None
        self.combined_image=None

    def erode_vertical_lines(self):
        vertical_kernel = np.array([[1], [1], [1], [1], [1], [1]])
        self.vertical_lines_eroded_image = cv2.erode(self.inverted_image, vertical_kernel, iterations=10)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, vertical_kernel, iterations=13)
    def erode_horizontal_lines(self):
      hor = np.array([[1,1,1,1,1,1]])
      self.horizontal_lines_eroded_image = cv2.erode(self.inverted_image, hor, iterations=10)
      self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, hor, iterations=13)
    def combine_eroded_images(self):
      self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)
