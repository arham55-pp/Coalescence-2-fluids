from pyoomph import *
from pyoomph.expressions import *
from lubrication_clean import LubricationEquations

class DropletCoalescence(Problem):
	def __init__(self):
		super(DropletCoalescence, self).__init__()
		# Geometry (see paper §2.1)
		self.L = 1                      # contact line radius (length scale)
		self.theta = 20 * pi / 180      # contact angle (20° as in paper)
		self.R = self.L / sin(self.theta)  # sphere radius
		self.H = self.R - self.L / tan(self.theta)  # apex height
		self.hp = 1e-4                  # precursor film thickness h_∞/L = 10⁻⁴
		self.Lx = 6                     # domain size [-3, 3]
		self.N = 1000                   # number of elements
		self.max_refinement_level = 6

		# Physical parameters
		self.sigma = 1  # surface tension (dimensionless)

	def define_problem(self):
		# 1D mesh from x = -3 to x = 3
		self.add_mesh(LineMesh(minimum=-3, size=self.Lx, N=self.N))

		h = var("h")

		# Clean lubrication equations (no surfactant, no disjoining pressure)
		eqs = LubricationEquations(sigma=self.sigma)
		eqs += MeshFileOutput()
		eqs += TextFileOutput()

		# Spherical cap height profile for two droplets
		# Droplet centers at x = ±sqrt(2RH - H^2)
		x_center = (2 * self.R * self.H - self.H**2)**(0.5)
		h1 = -self.R + self.H + (self.R**2 - (var("coordinate_x") + x_center)**2)**(0.5)
		h2 = -self.R + self.H + (self.R**2 - (var("coordinate_x") - x_center)**2)**(0.5)
		h_init = maximum(maximum(h1, h2), self.hp)

		eqs += InitialCondition(h=h_init)
		eqs += SpatialErrorEstimator(h=1)

		self.add_equations(eqs @ "domain")

if __name__ == "__main__":
	with DropletCoalescence() as problem:
		problem.run(100, outstep=0.1, startstep=0.001, maxstep=50,
		            temporal_error=1, spatial_adapt=1)
