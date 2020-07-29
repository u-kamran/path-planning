import numpy as np

from geometry import Point, Line, Polygon


class Grid:
    def __init__(self, minimum, maximum, first, final):
        self.minimum = minimum
        self.maximum = maximum
        self.width = maximum.x - minimum.x
        self.height = maximum.y - minimum.y
        self.first = first
        self.final = final

    def generateBoundaries(self, size):
        return [
            Polygon(self.width, size, 0.0, Point(self.minimum.x, self.maximum.y)),
            Polygon(self.width, size, 0.0, Point(self.minimum.x, self.minimum.y - size)),
            Polygon(size, self.height + size * 2.0, 0.0, Point(self.maximum.x, self.minimum.y - size)),
            Polygon(size, self.height + size * 2.0, 0.0, Point(self.minimum.x - size, self.minimum.y - size))
        ]

    def random(self, offset, size):
        return Point(
            np.random.randint(self.minimum.x+offset.x, self.maximum.x-offset.x-size.x+1),
            np.random.randint(self.minimum.y+offset.y, self.maximum.y-offset.y-size.y+1),
        )

    def generateObstacles(self, count, size, theta):
        obstacles = []
        for _ in range(count):
            obstacle = Polygon(size.x, size.y, theta, self.random(Point(0, 0), size))
            while self.first.intersects(obstacle) or self.final.intersects(obstacle):
                obstacle = Polygon(size.x, size.y, theta, self.random(Point(0, 0), size))
            obstacles.append(obstacle)
        return obstacles


def generate(inputs):

    grid = Grid(inputs["gridMinimum"], inputs["gridMaximum"], inputs["vehicleFirst"], inputs["vehicleFinal"])

    objectSize = inputs["objectSize"]

    boundaries = grid.generateBoundaries(min(objectSize.x, objectSize.y))

    obstacles = grid.generateObstacles(inputs["obstacleCount"], objectSize, inputs["obstacleTheta"])

    return grid, boundaries, obstacles
