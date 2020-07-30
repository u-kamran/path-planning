import numpy as np

from geometry import Point, Polygon


class Grid:
    def __init__(self, inputs):
        self.minimum = inputs["gridMinimum"]
        self.maximum = inputs["gridMaximum"]
        self.first = inputs["vehicleFirst"]
        self.final = inputs["vehicleFinal"]
        self.size = inputs["objectSize"]
        self.width = self.maximum.x - self.minimum.x
        self.height = self.maximum.y - self.minimum.y

    def generateBoundaries(self):
        dimension = min(self.size.x, self.size.y)
        return [
            Polygon(self.width, dimension, 0.0, Point(self.minimum.x, self.maximum.y)),
            Polygon(self.width, dimension, 0.0, Point(self.minimum.x, self.minimum.y - dimension)),
            Polygon(dimension, self.height + dimension * 2.0, 0.0, Point(self.maximum.x, self.minimum.y - dimension)),
            Polygon(dimension, self.height + dimension * 2.0, 0.0, Point(self.minimum.x - dimension, self.minimum.y - dimension))
        ]

    def random(self, offset, size):
        return Point(
            np.random.randint(self.minimum.x+offset.x, self.maximum.x-offset.x-size.x+1),
            np.random.randint(self.minimum.y+offset.y, self.maximum.y-offset.y-size.y+1),
        )

    def generateObstacles(self, count, theta):
        obstacles = []
        for _ in range(count):
            obstacle = Polygon(self.size.x, self.size.y, theta, self.random(Point(0, 0), self.size))
            while self.first.intersects(obstacle) or self.final.intersects(obstacle):
                obstacle = Polygon(self.size.x, self.size.y, theta, self.random(Point(0, 0), self.size))
            obstacles.append(obstacle)
        return obstacles


def generate(inputs):

    grid = Grid(inputs)

    boundaries = grid.generateBoundaries()

    obstacles = grid.generateObstacles(inputs["obstacleCount"], inputs["obstacleTheta"])

    return grid, boundaries, obstacles
