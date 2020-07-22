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
        ix = self.dx / (d+1)
        iy = self.dy / (d+1)
        points = []
        for i in range(1, d+1):
            points.append(Point(
                self.first.x+ix*i,
                self.first.y+iy*i
            ))
        points.append(self.final)
        return points


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


class Path:
    def __init__(self, points, obstacles, shortest):
        self.score = np.inf
        self.points = points
        self.fitness(obstacles, shortest)

    def fitness(self, obstacles, shortest):
        distance = 0
        collisions = 0

        for i in range(len(self.points) - 1):
            segment = Line(self.points[i], self.points[i + 1])
            distance += segment.length
            for obstacle in obstacles:
                if segment.intersects(obstacle):
                    collisions += 1000

        self.score = np.sqrt((distance / shortest.length) ** 2 + collisions ** 2)

        return self.score


def individual(grid, interpolation, segments, size):
    points = [grid.first]

    for s in range(segments):
        nextPoint = grid.random(size, Point(0, 0)) if s < segments-1 else grid.final
        segment = Line(points[-1], nextPoint)
        points.extend(segment.divide(interpolation))

    return points


def fitness(population, obstacles, shortest):
    individuals = []

    for path in population:
        distance = 0
        collisions = 0

        for i in range(len(path)-1):
            segment = Line(path[i], path[i+1])
            distance += segment.length
            for obstacle in obstacles:
                if segment.intersects(obstacle):
                    collisions += 1000

        score = np.sqrt((distance / shortest.length) ** 2 + collisions ** 2)

        individuals.append((score, path))

    return individuals


def sort(population):
    return sorted(population, key=lambda q: q[0])


def evolve(population, grid, size, count, chance):
    children = []
    while len(children) < count:
        parentA = np.random.randint(0, len(population))
        parentB = np.random.randint(0, len(population))
        if parentA != parentB:
            pathA = population[parentA][1]
            pathB = population[parentB][1]
            crossoverPosition = len(pathA) // 2
            child = pathA[:crossoverPosition] + pathB[crossoverPosition:]
            if np.random.random() <= chance:
                mutationPosition = np.random.randint(0, len(child))
                child[mutationPosition] = grid.random(size, Point(0, 0))
            children.append(child)
    return children


def visualize(grid, boundaries, obstacles, title, population=None):
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

    if population is not None:
        for path in population:
            px = [point.x for point in path]
            py = [point.y for point in path]
            ax.plot(px, py, 'y-', alpha=0.2, markersize=4)

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

    startTime = time.time()

    shortestPath = Line(grid.first, grid.final)

    populationCount = 80
    interpolation = 8
    pathSegments = 4

    initialPopulation = [individual(grid, interpolation, pathSegments, objectSize) for _ in range(populationCount)]

    gradedPopulation = sort(fitness(initialPopulation, obstacles, shortestPath))

    mutationChance = 0.04

    evolvedPopulation = evolve(gradedPopulation, grid, objectSize, populationCount, mutationChance)

    endTime = time.time()

    print("Time Elapsed:", endTime - startTime)

    visualize(grid, boundaries, obstacles, "Initial Population", initialPopulation)

    visualize(grid, boundaries, obstacles, "Evolved Population", evolvedPopulation)

    # input("Press Enter to Exit")


if __name__ == "__main__":
    main()
