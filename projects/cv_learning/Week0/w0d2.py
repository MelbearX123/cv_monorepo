from dataclasses import dataclass

# 1. Write a BoundingBox class: fields x1,y1,x2,y2. Methods: area(), aspect_ratio(), iou(other), __repr__.
# This class will appear in your final project.


class BoundingBox(x1, x2, y1, y2):
    def __int__(self, x1, x2, y1, y2):
        # assume x1 < x2, y1 < y2
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __repr__(self):
        print(f"x1={self.x1}, x2={self.x2}, y1={self.y1}, y2={self.y2}")

    def area(self):
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return abs(width * height)

    def aspect_ratio(self):
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return abs(width / height)

    def iou(self, other):
        # iou is area of intersection / area of union

        # Find area of intersection
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)

        width_inter = max(0, x2_inter - x1_inter)
        height_inter = max(0, y2_inter - y1_inter)

        intersection = width_inter * height_inter

        # Find area of union
        union = self.area() + other.area() - intersection

        if union == 0:
            return 0.0

        return intersection / union


# 2. Write a Detection dataclass with fields: box (BoundingBox), confidence (float), class_id (int), class_name (str).
@dataclass
class Detection:
    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str


# Write a base class Pipeline with a process(frame) method that raises NotImplementedError.
#  Subclass it with a GrayscalePipeline that implements process().

# Instantiate 10 BoundingBox objects, put them in a list, sort by area using a lambda.

# Add a class attribute instance_count to BoundingBox that increments on __init__ — verify it counts correctly.

# Practice: given someone else's class (find any OpenCV tracker class online), read it and explain
# what each method does out loud.
