import numpy as np
import matplotlib.pyplot as plt 


def scaleTrack(track, boundX, boundY):
    minX = track[0][0]
    maxX = track[0][0]
    minY = track[0][1]
    maxY = track[0][1]

    for point in track:
        minX = min(minX, point[0])
        maxX = max(maxX, point[0])
        minY = min(minY, point[1])
        maxY = max(maxY, point[1])

    scaleX = boundX / (maxX - minX)
    scaleY = boundY / (maxY - minY)
    shiftX = -(minX + maxX) / 2
    shiftY = -(minY + maxY) / 2

    for point in track:
        point[0] = (point[0] + shiftX) * scaleX
        point[1] = (point[1] + shiftY) * scaleY

    firstPointX = track[0][0]
    firstPointY = track[0][1]

    for point in track:
        point[0] -= firstPointX
        point[1] -= firstPointY

    return track

def plotTrack(pop, sampleRate):

    for j in range(0, min(len(pop), 5)):
        acc = pop[j]
        acc = np.vstack((acc, completeAcc(acc)))
        poly = polysFromAccs(acc)
        bez = bezFromPolys(poly)
        smpl = sampleBezSeries(bez, sampleRate)

        plt.figure(j)

        for i in range(0, len(acc)):
            subPlot = smpl.transpose()[sampleRate * i : sampleRate * (i + 1) - 1].transpose()
            plt.plot(subPlot[0], subPlot[1])
            plt.axis('equal')
    
    plt.show()


class TrackFeatures:
    def __init__(self, pos1, vel1, acc1, acc2):
        self.p1 = pos1
        self.v1 = vel1
        self.a1 = acc1
        self.a2 = acc2


class CubicPoly:
    def __init__(self, trackFeatures):
        M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 / 2, 0], [0, 0, -1 / 6, 1 / 6]])
        p1 = trackFeatures.p1
        v1 = trackFeatures.v1
        a1 = trackFeatures.a1
        a2 = trackFeatures.a2

        self.coeffX = M.dot(np.array([p1[0], v1[0], a1[0], a2[0]]))
        self.coeffY = M.dot(np.array([p1[1], v1[1], a1[1], a2[1]]))

    def startPos(self):
        return np.array([self.coeffX[0], self.coeffY[0]])

    def endPos(self):
        N = np.array([1, 1, 1, 1])
        return np.array([N.dot(self.coeffX), N.dot(self.coeffY)])

    def startAcc(self):
        return np.array([2 * self.coeffX[2], 2 * self.coeffY[2]])

    def startVel(self):
        L = np.array([0, 1, 0, 0])
        return np.array([L.dot(self.coeffX), L.dot(self.coeffY)])

    def endVel(self):
        L = np.array([0, 1, 2, 3])
        return np.array([L.dot(self.coeffX), L.dot(self.coeffY)])

    def endAcc(self):
        K = np.array([0, 0, 2, 6])
        return np.array([K.dot(self.coeffX), K.dot(self.coeffY)])

    def pos(self, t):
        return [self.coeffX[3] * t * t * t + self.coeffX[2] * t * t + self.coeffX[1] * t + self.coeffX[0],
                self.coeffY[3] * t * t * t + self.coeffY[2] * t * t + self.coeffY[1] * t + self.coeffY[0]]

    def vel(self, t):
        return [3 * self.coeffX[3] * t * t + 2 * self.coeffX[2] * t + self.coeffX[1],
                3 * self.coeffY[3] * t * t + 2 * self.coeffY[2] * t + self.coeffY[1]]

    def acc(self, t):
        return [6 * self.coeffX[3] * t + 2 * self.coeffX[2],
                6 * self.coeffY[3] * t + 2 * self.coeffY[2]]


class CubicBez:
    def __init__(self, cubicPoly):
          B = np.array([[1, 0, 0, 0], [1, 1 / 3, 0, 0], [1, 2 / 3, 1 / 3, 0], [1, 1, 1, 1]])
          x = B.dot(cubicPoly.coeffX)
          y = B.dot(cubicPoly.coeffY)
          self.points = np.concatenate((x, y)).reshape(2, -1)

def bezFromPolys(polys):
    bez = []
    for p in polys:
        bez.append(CubicBez(p))
    return bez

def polysFromAccs(accs):

    posStart = np.array([0, 0])
    velStart = initVel(accs)
    
    accStart = accs[0]
    accEnd = accs[1]

    fs = [TrackFeatures(posStart, velStart, accStart, accEnd)]
    ps = [CubicPoly(fs[0])]

    for i in range(1, len(accs)):
        polyBefore = ps[i - 1]

        posStart = polyBefore.endPos()
        velStart = polyBefore.endVel()

        accStart = accs[i]
        accEnd = accs[i + 1] if i < len(accs) - 1 else accs[0]

        f = TrackFeatures(posStart, velStart, accStart, accEnd)
        fs.append(f)
        ps.append(CubicPoly(f))

    return ps

def initVel(accs):
    n = len(accs)
    T = -np.array(range(n, 0, -1)) / n
    T[0] = -1 / 2
    return T.dot(accs)

def doBezIntersect(bez, delta):

    points = []

    for b in bez:
        ps = b.points.transpose()
        for i in range(0, len(ps) - 1):
            points.append(ps[i])

    n = len(points)
    
    for i in range(0, n - 2):
        for j in range(i + 2, n if i != 0 else n - 1):           
            p1 = points[i]
            p2 = points[i + 1]
            q1 = points[j]
            q2 = points[(j + 1) % n]

            if doLinesIntersectDelta(p1, p2, q1, q2, delta): 
                return True

    return False


def doLinesIntersectDelta(p1, p2, q1, q2, delta):
    if doLinesIntersect(p1, p2, q1, q2): return True
    if minDist(p1, np.array([q1, q2])) <= delta: return True
    if minDist(p2, np.array([q1, q2])) <= delta: return True
    if minDist(q1, np.array([p1, p2])) <= delta: return True
    if minDist(q2, np.array([p1, p2])) <= delta: return True
    return False


def doLinesIntersect(p1, p2, q1, q2):
    r = p2[0] - p1[0]
    s = q1[0] - q2[0]
    t = p2[1] - p1[1]
    u = q1[1] - q2[1]
    v = q1[0] - p1[0]
    w = q1[1] - p1[1]
    d0 = r * u - s * t
    if d0 == 0:         return False
    d1 = v * u - s * w
    d2 = r * w - v * t
    i1 = d1 / d0
    i2 = d2 / d0
    return (i1 >= 0) and (i1 <= 1) and (i2 >= 0) and (i2 <= 1)


def minDist(p, l):
    a = l[1] - l[0]
    b = l[0]
    param = a.transpose().dot(p - b) / a.transpose().dot(a)
    if param > 1: param = 1
    if param < 0: param = 0
    return np.linalg.norm(param * a + b - p)


def containsDuplicate(indiv, pop, threshold):
    isDup = False
    for p in pop:
        if(isDuplicate(indiv, p, threshold)):
            isDup = True
            break           

    return isDup

def isDuplicate(a, b, threshold):

    inCommon = 0

    for i in range(0, len(a)):
        if np.linalg.norm(a[i] - b[i]) < threshold: 
            inCommon += 1    

    return inCommon > 0.2 * len(a)


def sampleBezSeries(bez, timesPerBez):

    basis = []
    
    for i in range(0, timesPerBez):
        x = i / timesPerBez
        
        b1 = x ** 3
        b2 = 3 * (x ** 2) * (1 - x)
        b3 = 3 * x * (1 - x) ** 2
        b4 = (1 - x) ** 3
        
        basis.append([b4, b3, b2, b1])

    smpls = None

    for b in bez:
        smplBez = b.points.dot(np.array(basis).transpose())            

        if smpls is None:
            smpls = smplBez
        else:
            smpls = np.hstack((smpls, smplBez))

    return smpls


def completeAcc(accs):
    return -sum(accs)


def rotate(x, by, rev=False):  
    if np.linalg.norm(by) == 0:
        return x

    return rotMatr(by, rev).dot(x)


def rotMatr(x, rev=False):    
  x = x / np.linalg.norm(x)
  return np.array([[x[0], x[1]], [-x[1], x[0]]] if rev else [[x[0], -x[1]], [x[1], x[0]]])
  