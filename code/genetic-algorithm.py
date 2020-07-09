import time
import random

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
    def __init__(self, minimum, maximum, offset):
        self.minimum = minimum
        self.maximum = maximum
        self.width = maximum.x - minimum.x
        self.height = maximum.y - minimum.y
        self.first = self.minimum.translate(Point(offset, offset))
        self.final = self.maximum.translate(Point(-offset, -offset))

    def random(self, limit):
        return Point(
            np.random.randint(self.minimum.x+limit, self.maximum.x-limit),
            np.random.randint(self.minimum.y+limit, self.maximum.y-limit),
        )

    def borders(self, size):
        return [
            Polygon(self.width, size, 0.0, Point(self.minimum.x, self.maximum.y)),
            Polygon(self.width, size, 0.0, Point(self.minimum.x, self.minimum.y - size)),
            Polygon(size, self.height + size * 2.0, 0.0, Point(self.maximum.x, self.minimum.y - size)),
            Polygon(size, self.height + size * 2.0, 0.0, Point(self.minimum.x - size, self.minimum.y - size))
        ]

    def generate(self, sx, sy, theta, limit):
        obstacle = Polygon(sx, sy, theta, self.random(limit))
        while self.first.intersects(obstacle) or self.final.intersects(obstacle):
            obstacle = Polygon(sx, sy, theta, self.random(limit))
        return obstacle


def main():
    vehicleOffset = 2.0
    vehicleSize = 1.0

    environmentMin = 0.0
    environmentMax = 20.0

    vehicleGrid = Grid(Point(environmentMin, environmentMin), Point(environmentMax, environmentMax), vehicleOffset)

    shortestPath = Line(vehicleGrid.first, vehicleGrid.final)

    boundaries = vehicleGrid.borders(vehicleSize)

    obstacles = [vehicleGrid.generate(vehicleSize, vehicleSize, 0.0, 0.0) for _ in range(40)]

    populationCount = 80

    # work in progress...

    # input("Press Enter to Exit")


if __name__ == "__main__":
    main()
