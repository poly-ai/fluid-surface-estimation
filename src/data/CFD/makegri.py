from math import ceil, sqrt
from random import  uniform

import numpy as np
from readgri import writegri

minX = 0
minY = 0
maxX = 1
maxY = 1

unitX = 0.01
unitY = sqrt(0.75) * unitX

vertices = []
interior = []
exterior = []
elements = []

xRange = ceil(maxX / unitX)
yRange = ceil(maxY / unitY / 2)

for y in range(0, yRange):
    for x in range(0, xRange):
        a = [x * unitX, y * unitY * 2]
        b = [(x + 1) * unitX, y * unitY * 2]

        c = [(x - 0.5) * unitX, (y + 0.5) * unitY * 2]
        d = [(x + 0.5) * unitX, (y + 0.5) * unitY * 2]

        e = [x * unitX, (y + 1) * unitY * 2]
        f = [(x + 1) * unitX, (y + 1) * unitY * 2]

        i0 = len(vertices)
        i1 = len(vertices) + 2
        i2 = len(vertices) + 1
        i3 = len(vertices) + 3

        i4 = len(vertices) + xRange * 2 + 3
        i5 = len(vertices) + xRange * 2 + 5

        if y + 1 == yRange:
            i4 -= x
            i5 -= x + 1

        offset = unitX * 0.3 * uniform(-1, 1)
        if y == 0:
            vertices.append([minX + x * unitX, y * unitY * 2])
        else:
            vertices.append([minX + x * unitX, y * unitY * 2 + offset])

        if x == 0:
            vertices.append([minX, (y + 0.5) * unitY * 2 + offset])

            exterior.append([i2, i0, len(elements), 0])
            exterior.append([i4, i2, len(elements) + 2, 0])
        else:
            vertices.append([minX + (x - 0.5) * unitX, (y + 0.5) * unitY * 2 + offset])

            interior.append([i2, i0, len(elements), len(elements) - 3])
            interior.append([i4, i2, len(elements) + 2, len(elements) - 1])

        interior.append([i0, i3, len(elements), len(elements) + 1])
        interior.append([i3, i2, len(elements), len(elements) + 2])

        if y == 0:
            exterior.append([i1, i0, len(elements) + 1, 0])
        else:
            interior.append([i1, i0, len(elements) + 1, len(elements) - 402])

        interior.append([i4, i3, len(elements) + 2, len(elements) + 3])

        if y + 1 == yRange:
            exterior.append([i4, i5, len(elements) + 3, 0])

        elements.append([i0, i2, i3])
        elements.append([i0, i3, i1])
        elements.append([i2, i4, i3])
        elements.append([i3, i4, i5])

    i0 = len(vertices)
    i1 = len(vertices) + 1
    i2 = len(vertices) + 2
    i3 = len(vertices) + xRange * 2 + 3

    if y + 1 == yRange:
        i3 -= xRange

    vertices.append([maxX, y * unitY * 2])
    vertices.append([maxX - 0.5 * unitX, (y + 0.5) * unitY * 2])
    vertices.append([maxX, (y + 0.5) * unitY * 2])

    interior.append([i1, i0, len(elements), len(elements) - 3])
    exterior.append([i0, i2, len(elements), 0])
    interior.append([i1, i2, len(elements), len(elements) + 1])

    interior.append([i3, i1, len(elements) + 1, len(elements)])
    exterior.append([i3, i2, len(elements) + 1, 0])

    elements.append([i0, i1, i2])
    elements.append([i1, i3, i2])

for x in range(0, xRange):
    vertices.append([minX + x * unitX, maxY])

vertices.append([maxX, maxY])

mesh = {
    "V": np.array(vertices),
    "E": np.array(elements),
    "IE": np.array(interior),
    "BE": np.array(exterior),
    "Bname": np.array(["WALL"]),
}

writegri(mesh, "max.gri")
