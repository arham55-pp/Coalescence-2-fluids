import sys
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.meshdatacache import MeshDataEigenModes
from pyoomph.typings import List, Optional, Union
from lubrication import LubricationEquations  # surfactant version (beta, Pe)
from pyoomph.output.plotting import *
from pyoomph.output.meshio import TextFileOutputAlongLine


from pyoomph.expressions.units import *  # units
from pyoomph.expressions.phys_consts import gas_constant  # and the gas constant

class PlotterTry(MatplotlibPlotter):
	def __init__(self, problem: Problem, filetrunk: str = "plot_{:05d}", fileext: str | List[str] = "png", eigenvector: int | None = None, eigenmode: MeshDataEigenModes = "abs", add_eigen_to_mesh_positions: bool = True, position_eigen_scale: float = 1):
		super().__init__(problem, filetrunk, fileext, eigenvector, eigenmode, add_eigen_to_mesh_positions, position_eigen_scale)

	def define_plot(self):
		p = self.get_problem()
		colorbar_1 = self.add_colorbar("h", cmap ='seismic', position = 'top center')

		self.set_view(-3, 0, 0 , 2.05)

		self.add_plot("domain/h", colorbar=colorbar_1)

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

		# Surfactant parameters (with defaults, see paper §2.2)
		self.beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1    # surfactant strength (β)
		self.Pe = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0      # Péclet number
		self.Gamma_0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8 # initial surfactant concentration
		# self.plotter = PlotterTry(self)
			
					
	def define_problem(self):
		self.add_mesh(LineMesh(minimum=-3, size=self.Lx, N=self.N)) 
		
		h=var("h") 
		Gamma=var("Gamma")

		# self.sigma=self.sigma-self.ma*Gamma

		eqs=LubricationEquations(beta=self.beta, Pe=self.Pe) # equations
		eqs+=MeshFileOutput() # output	
		x=var("coordinate")

		h1=-self.R + self.H + (self.R**2 - (var("coordinate_x") + (2 * self.R * self.H - self.H**2)**(0.5))**2)**(0.5)
		h2=-self.R + self.H + (self.R**2 - (var("coordinate_x") - (2 * self.R * self.H - self.H**2)**(0.5))**2)**(0.5)
		h_init=h_init=maximum(maximum(h1,h2),self.hp) 

		# Surfactant IC: Γ = Γ₀ on left droplet (x<0), Γ = 0 on right droplet (x>0)
		Gamma_init = self.Gamma_0 * (0.5 - 0.5*tanh(var("coordinate_x") / self.hp))
		
		eqs+=InitialCondition(Gamma=Gamma_init)
		eqs+=InitialCondition(h=h_init) 
		
		eqs+=SpatialErrorEstimator(h=1) # refine based on the height field
		eqs += TextFileOutput()

		# self+=TextFileOutputAlongLine(filename='profile', start=(-3,0), end=(3,0), N =200) @ "domain"
		
		self.add_equations(eqs@"domain") # adding the equation
		
if __name__=="__main__":
	with DropletCoalescence() as problem:
		problem.run(100,outstep=0.1,startstep=0.01,maxstep=50,temporal_error=1,spatial_adapt=1)
