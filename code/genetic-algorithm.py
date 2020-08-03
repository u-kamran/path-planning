import sys
import time

import numpy as np

from scipy.special import comb

import matplotlib.pyplot as plt
import matplotlib.patches as ptc

import common.environment as env
import common.inputs as inputs

from common.geometry import Point, Line


class Path:
    def __init__(self, points):
        self.score = np.inf
        self.points = points

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


def individual(grid, interpolation, segments):
    points = [grid.first]

    for s in range(segments):
        nextPoint = grid.random(grid.size, Point(0, 0)) if s < segments-1 else grid.final
        segment = Line(points[-1], nextPoint)
        points.extend(segment.divide(interpolation))

    return points


def bezierCurve(points, samples):
    t = np.linspace(0.0, 1.0, samples)

    px = [p.x for p in points]
    py = [p.y for p in points]

    polynomials = [bernsteinPolynomial(len(points) - 1, i, t) for i in range(len(points))]

    vx = np.dot(px, polynomials)
    vy = np.dot(py, polynomials)

    return [Point(vx[s], vy[s]) for s in range(samples)]


def bernsteinPolynomial(n, i, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def sort(population):
    if len(population) <= 1:
        return population

    mid = len(population) // 2

    left = sort(population[:mid])
    right = sort(population[mid:])

    return merge(left, right, population.copy())


def merge(left, right, population):
    leftPosition = 0
    rightPosition = 0

    while leftPosition < len(left) and rightPosition < len(right):

        if left[leftPosition].score <= right[rightPosition].score:
            population[leftPosition + rightPosition] = left[leftPosition]
            leftPosition += 1
        else:
            population[leftPosition + rightPosition] = right[rightPosition]
            rightPosition += 1

    for leftPosition in range(leftPosition, len(left)):
        population[leftPosition + rightPosition] = left[leftPosition]

    for rightPosition in range(rightPosition, len(right)):
        population[leftPosition + rightPosition] = right[rightPosition]

    return population


def evolve(population, grid, count, chance):
    children = []
    while len(children) < count:
        parentA = np.random.randint(0, len(population))
        parentB = np.random.randint(0, len(population))
        if parentA != parentB:
            pathA = population[parentA].points
            pathB = population[parentB].points
            crossoverPosition = len(pathA) // 2
            child = pathA[:crossoverPosition] + pathB[crossoverPosition:]
            if np.random.random() <= chance:
                mutationPosition = np.random.randint(0, len(child))
                child[mutationPosition] = grid.random(grid.size, Point(0, 0))
            children.append(child)
    return children


def select(graded, evolved, count):
    graded.extend(evolved)
    graded = sort(graded)

    # truncation selection
    if len(graded) > count:
        graded = graded[:count]

    return graded


def visualize(grid, boundaries, obstacles, title, population=None, optimal=None):
    fig, ax = plt.subplots()

    ax.set_title(title, weight='bold')

    ax.annotate(
        "FIRST", (grid.first.x, grid.minimum.y - 0.1),
        horizontalalignment='center',
        verticalalignment='top',
        weight='bold', color='w',
    )

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

    if population is not None:
        for path in population:
            px = [point.x for point in path.points]
            py = [point.y for point in path.points]
            ax.plot(px, py, 'y-', alpha=0.2, markersize=4)

    if optimal is not None:
        px = [point.x for point in optimal.points]
        py = [point.y for point in optimal.points]
        ax.plot(px, py, 'c-', alpha=0.8, markersize=4)

    ax.plot(grid.first.x, grid.first.y, 'co')
    ax.plot(grid.final.x, grid.final.y, 'mo')

    ax.grid()

    plt.axis('scaled')
    plt.show()


def scatterPlot(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.scatter(x, y, marker='o', c='c')
    plt.grid()
    plt.show()


def main(args=None):

    filename = "genetic-algorithm"

    if args is None:
        args = sys.argv[1:]

    arguments = inputs.parse(filename, args)

    grid, boundaries, obstacles = env.generate(arguments)

    visualize(grid, boundaries, obstacles, "Environment")

    startTime = time.time()

    shortestPath = Line(grid.first, grid.final)

    initialPopulation = []

    for _ in range(arguments["populationCount"]):
        path = Path(individual(grid, arguments["interpolation"], arguments["pathSegments"]))
        path.points = bezierCurve(path.points, arguments["curveSamples"])
        path.fitness(obstacles, shortestPath)
        initialPopulation.append(path)

    gradedPopulation = sort(initialPopulation)

    finalPopulation = None
    optimalPath = None

    averageFitness = []
    evolutionCount = 0

    while evolutionCount < arguments["evolutionMax"]:
        evolvedPaths = evolve(gradedPopulation, grid, arguments["populationCount"], arguments["mutationChance"])

        evolvedPopulation = []

        for points in evolvedPaths:
            path = Path(points)
            path.fitness(obstacles, shortestPath)
            evolvedPopulation.append(path)

        gradedPopulation = select(gradedPopulation, evolvedPopulation, arguments["populationCount"])

        average = 0
        for path in gradedPopulation:
            average += path.score
        average /= len(gradedPopulation)

        averageFitness.append(average)

        print(
            "Evolution:", evolutionCount + 1,
            "| Average Fitness:", average,
            "| Best Fitness Value:", gradedPopulation[0].score
        )

        evolutionCount += 1

        if evolutionCount == arguments["evolutionMax"]:
            finalPopulation = gradedPopulation
            optimalPath = gradedPopulation[0]

    endTime = time.time()

    print("Time Elapsed:", endTime - startTime)

    visualize(grid, boundaries, obstacles, "Initial Population", initialPopulation)

    visualize(grid, boundaries, obstacles, "Final Population", finalPopulation)

    visualize(grid, boundaries, obstacles, "Optimal Path", None, optimalPath)

    scatterPlot(
        np.arange(1, arguments["evolutionMax"] + 1), averageFitness,
        "Average Fitness of Population", "Evolution", "Fitness Value"
    )

    # input("Press Enter to Exit")


if __name__ == "__main__":
    main()
