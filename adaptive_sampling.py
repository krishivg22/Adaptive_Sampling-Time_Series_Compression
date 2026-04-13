import numpy as np
import matplotlib.pyplot as plt
class Learner1D:
    def __init__(self, f, bounds, tol=1e-3):
        self.f = f
        self.xmin, self.xmax = bounds
        self.data = {}  # x -> y
        self.tol = tol
        
        # start with endpoints
        self.tell(self.xmin, f(self.xmin))
        self.tell(self.xmax, f(self.xmax))

    def tell(self, x, y):
        self.data[x] = y

    def ask(self):
        xs = sorted(self.data.keys())
        
        best_error = -1
        best_mid = None
        
        for i in range(len(xs)-1):
            x1, x2 = xs[i], xs[i+1]
            y1, y2 = self.data[x1], self.data[x2]
            
            mid = 0.5*(x1 + x2)
            pred = 0.5*(y1 + y2)
            actual = self.f(mid)
            err = abs(actual - pred)
            
            if err > best_error:
                best_error = err
                best_mid = mid
        
        return best_mid, best_error
    
    def should_stop(self, error):
        return error < self.tol
    
    def run(self, max_steps=200):
        for _ in range(max_steps):
            x, err = self.ask()
            if self.should_stop(err):
                break
            self.tell(x, self.f(x))
        return self.data



class Learner2D:
    def __init__(self, f, bounds_x, bounds_y, tol=1e-2):
        self.f = f
        self.xmin, self.xmax = bounds_x
        self.ymin, self.ymax = bounds_y
        self.tol = tol
        self.points = {}  # (x, y) -> f(x,y)

        # initial grid corners only
        for x in [self.xmin, self.xmax]:
            for y in [self.ymin, self.ymax]:
                self.tell((x, y), f((x, y)))

    def tell(self, xy, z):
        self.points[tuple(xy)] = z

    def ask(self):
        best_err = -1
        best_xy = None
        
        pts = list(self.points.items())
        xs = sorted(set([p[0][0] for p in pts]))
        ys = sorted(set([p[0][1] for p in pts]))

        for i in range(len(xs)-1):
            for j in range(len(ys)-1):
                x1, x2 = xs[i], xs[i+1]
                y1, y2 = ys[j], ys[j+1]

                # Check if all 4 corners exist
                corners = [
                    (x1, y1),
                    (x1, y2),
                    (x2, y1),
                    (x2, y2)
                ]

                if not all(c in self.points for c in corners):
                    continue  # skip incomplete rectangles

                # midpoint
                mid = (0.5*(x1+x2), 0.5*(y1+y2))
                z_mid = self.f(mid)

                neighbor_vals = [self.points[c] for c in corners]
                pred = np.mean(neighbor_vals)
                err = abs(pred - z_mid)

                if err > best_err:
                    best_err = err
                    best_xy = mid
        
        return best_xy, best_err
    
    def should_stop(self, error):
        if error is None:
            return True
        return error < self.tol
    
    def run(self, max_steps=300):
        for _ in range(max_steps):
            xy, err = self.ask()

            # no valid midpoint found = stop
            if xy is None or self.should_stop(err):
                break

            self.tell(xy, self.f(xy))

        return self.points



class DiscreteLearner:
    def __init__(self, f, points):
        self.f = f
        self.points = list(points)
        self.data = {}
        self.unseen = set(points)

    def ask(self):
        # simple: choose any unseen point
        return next(iter(self.unseen))
    
    def tell(self, x, y):
        self.data[x] = y
        self.unseen.remove(x)

    def run(self):
        while self.unseen:
            x = self.ask()
            self.tell(x, self.f(x))
        return self.data


class SequenceLearner:
    def __init__(self, f, n, tol=1e-4):
        self.f = f
        self.n = n
        self.data = {}
        self.tol = tol
        
        # sample boundaries
        self.tell(0, f(0))
        self.tell(n-1, f(n-1))

    def tell(self, i, y):
        self.data[i] = y

    def ask(self):
        keys = sorted(self.data.keys())
        
        best_err = -1
        best_mid = None
        
        for k in range(len(keys)-1):
            i1, i2 = keys[k], keys[k+1]
            mid = (i1 + i2)//2
            if mid in self.data:
                continue
            
            pred = 0.5*(self.data[i1] + self.data[i2])
            actual = self.f(mid)
            err = abs(pred - actual)
            
            if err > best_err:
                best_err = err
                best_mid = mid
        
        return best_mid, best_err
    
    def should_stop(self, err):
        return err < self.tol
    
    def run(self, max_steps=500):
        for _ in range(max_steps):
            mid, err = self.ask()
            if mid is None or self.should_stop(err):
                break
            self.tell(mid, self.f(mid))
        return self.data


class BalancingLearner:
    def __init__(self, learners):
        self.learners = learners

    def ask(self):
        best_err = -1
        best_learner = None
        best_x = None

        for L in self.learners:
            x, err = L.ask()
            if err > best_err:
                best_err = err
                best_learner = L
                best_x = x
        
        return best_learner, best_x, best_err

    def run(self, steps=300):
        for _ in range(steps):
            L, x, err = self.ask()
            if L.should_stop(err):
                continue
            L.tell(x, L.f(x))



if __name__ == "__main__":
    # Example for Learner1D
    f1 = lambda x: np.sin(10*x) + x
    L1 = Learner1D(f1, (0, 1), tol=1e-3)
    data_1d = L1.run()
    print("1D learner samples:", len(data_1d))

    # Example for Learner2D
    f2 = lambda xy: np.sin(xy[0]*xy[1]*5)
    L2 = Learner2D(f2, (0, 1), (0, 1))
    data_2d = L2.run()
    print("2D learner samples:", len(data_2d))

    # Example for DiscreteLearner
    f3 = lambda x: x**2
    DL = DiscreteLearner(f3, points=range(10))
    data_disc = DL.run()
    print("Discrete learner samples:", len(data_disc))

    # Example for SequenceLearner
    f4 = lambda i: np.sqrt(i) + np.sin(i)
    SL = SequenceLearner(f4, n=50)
    seq_data = SL.run()
    print("Sequence learner samples:", len(seq_data))

    # Example for BalancingLearner (combine two 1D learners)
    L1a = Learner1D(lambda x: np.sin(15*x), (0,1))
    L1b = Learner1D(lambda x: np.cos(10*x), (0,1))
    BL = BalancingLearner([L1a, L1b])
    BL.run(steps=100)
    print("Balancing learner finished.")


# 1D visualization
def visualize_1d(data_1d, title="1D Adaptive Sampling"):
    xs = sorted(data_1d.keys())
    ys = [data_1d[x] for x in xs]

    plt.figure(figsize=(8,4))
    plt.plot(xs, ys, "-o", markersize=4)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

# 2D visualization
def visualize_2d(data_2d, title="2D Adaptive Sampling Points"):
    pts = np.array(list(data_2d.keys()))

    plt.figure(figsize=(6,6))
    plt.scatter(pts[:,0], pts[:,1], c="blue", s=30)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

# Discrete learner visualization (bar chart)
def visualize_discrete(data_disc, title="Discrete Adaptive Sampling"):
    xs = sorted(data_disc.keys())
    ys = [data_disc[x] for x in xs]

    plt.figure(figsize=(8,4))
    plt.bar(xs, ys, color="green")
    plt.title(title)
    plt.xlabel("Discrete Points")
    plt.ylabel("f(x)")
    plt.grid(True, axis="y")
    plt.show()

# Sequence learner visualization
def visualize_sequence(seq_data, title="Sequence Adaptive Sampling"):
    xs = sorted(seq_data.keys())
    ys = [seq_data[x] for x in xs]

    plt.figure(figsize=(8,4))
    plt.plot(xs, ys, "-o", markersize=4)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("f(index)")
    plt.grid(True)
    plt.show()


# Balancing learner (comparative visualization)
def visualize_balancing(learners, titles=None):
    n = len(learners)
    plt.figure(figsize=(6*n, 4))

    for i, L in enumerate(learners):
        xs = sorted(L.data.keys())
        ys = [L.data[x] for x in xs]
        plt.subplot(1, n, i+1)
        plt.plot(xs, ys, "-o", markersize=4)
        if titles:
            plt.title(titles[i])
        else:
            plt.title(f"Learner {i+1}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
    
    plt.show()

# 1D learner visualization
visualize_1d(data_1d, "1D Adaptive Sampling")

# 2D learner visualization
visualize_2d(data_2d, "2D Adaptive Sampling")

# Discrete learner visualization
visualize_discrete(data_disc, "Discrete Learner Sampling")

# Sequence learner visualization
visualize_sequence(seq_data, "Sequence Adaptive Sampling")

# Balancing learner visualization (compares L1a and L1b)
visualize_balancing(
    [L1a, L1b],
    titles=["Balanced Learner 1", "Balanced Learner 2"]
)
