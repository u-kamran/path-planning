import time

import numpy as np

from scipy.special import comb

import matplotlib.pyplot as plt
import matplotlib.patches as ptc

from shapely.geometry import Point as SPoint
from shapely.geometry import Polygon as SPolygon
from shapely.geometry import LineString as SLine


class Point(SPoint):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.center = (self.x, self.y)

    def scale(self, sx, sy):
        return Point(self.x * sx, self.y * sy)

    def rotate(self, theta):
        return Point(
            self.x * np.cos(theta) - self.y * np.sin(theta),
            self.x * np.sin(theta) + self.y * np.cos(theta)
        )

    def translate(self, point):
        return Point(self.x + point.x, self.y + point.y)


class Line(SLine):
    def __init__(self, first, final):
        super().__init__([first.center, final.center])
        self.first = first
        self.final = final
        self.dx = final.x - first.x
        self.dy = final.y - first.y
        self.theta = np.arctan2(self.dy, self.dx)

    def divide(self, d):
        ix = self.dx / d
        iy = self.dy / d
        segments = []
        previous = self.first
        for i in range(1, d):
            current = Point(self.first.x+ix*i, self.first.y+iy*i)
            segments.append(Line(previous, current))
            previous = current
        segments.append(Line(previous, self.final))
        return segments


class Polygon(SPolygon):
    def __init__(self, sx, sy, theta, t):
        corners = [
            Point(0.0, 0.0).scale(sx, sy).rotate(theta).translate(t).center,
            Point(1.0, 0.0).scale(sx, sy).rotate(theta).translate(t).center,
            Point(1.0, 1.0).scale(sx, sy).rotate(theta).translate(t).center,
            Point(0.0, 1.0).scale(sx, sy).rotate(theta).translate(t).center
        ]
        super().__init__(corners)
        self.datum = corners[0]
        self.width = sx
        self.height = sy
        self.rotation = theta

    def angle(self):
        # convert from radians to degrees
        return self.rotation * 180.0 / np.pi


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

    def random(self, size):
        return Point(
            np.random.randint(self.minimum.x, self.maximum.x-size.x+1),
            np.random.randint(self.minimum.y, self.maximum.y-size.y+1),
        )

    def generateObstacles(self, count, size, theta):
        obstacles = []
        for _ in range(count):
            obstacle = Polygon(size.x, size.y, theta, self.random(size))
            while self.first.intersects(obstacle) or self.final.intersects(obstacle):
                obstacle = Polygon(size.x, size.y, theta, self.random(size))
            obstacles.append(obstacle)
        return obstacles


def visualize(grid, boundaries, obstacles, title):
    fig, ax = plt.subplots()

    ax.set_title(title, weight='bold')

    ax.plot(grid.first.x, grid.first.y, 'co')

    ax.annotate(
        "FIRST", (grid.first.x, grid.minimum.y - 0.1),
        horizontalalignment='center',
        verticalalignment='top',
        weight='bold', color='w',
    )

    ax.plot(grid.final.x, grid.final.y, 'mo')

    ax.annotate(
        "FINAL", (grid.final.x, grid.maximum.y + 0.0),
        horizontalalignment='center',
        verticalalignment='bottom',
        weight='bold', color='w'
    )

    for obstacle in obstacles:
        rectangle = ptc.Rectangle(
            obstacle.datum,
            obstacle.width,
            obstacle.height,
            obstacle.angle(),
            edgecolor='None', facecolor='grey', alpha=1.0
        )
        ax.add_patch(rectangle)

    for boundary in boundaries:
        rectangle = ptc.Rectangle(
            boundary.datum,
            boundary.width,
            boundary.height,
            boundary.angle(),
            edgecolor='None', facecolor='black', alpha=1.0
        )
        ax.add_patch(rectangle)

    ax.grid()

    plt.axis('scaled')
    plt.show()


def main():
    gridMinimum = Point(0.0, 0.0)
    gridMaximum = Point(20.0, 20.0)

    vehicleFirst = Point(2.0, 2.0)
    vehicleFinal = Point(18.0, 18.0)

    grid = Grid(gridMinimum, gridMaximum, vehicleFirst, vehicleFinal)

    objectSize = Point(1.0, 1.0)

    boundaries = grid.generateBoundaries(min(objectSize.x, objectSize.y))

    obstacleCount = 40
    obstacleTheta = 0.0

    obstacles = grid.generateObstacles(obstacleCount, objectSize, obstacleTheta)

    visualize(grid, boundaries, obstacles, "Environment")

    shortestPath = Line(grid.first, grid.final)

    startTime = time.time()

    populationCount = 80
    # work in progress...

    endTime = time.time()

    print("Time Elapsed:", endTime - startTime)

    # input("Press Enter to Exit")


if __name__ == "__main__":
    main()
