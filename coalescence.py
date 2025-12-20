import argparse
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.meshdatacache import MeshDataEigenModes
from pyoomph.typings import List, Optional, Union
from lubrication import LubricationEquations  # surfactant version (beta, Pe)
from pyoomph.output.plotting import *
from pyoomph.output.meshio import TextFileOutputAlongLine


from pyoomph.expressions.units import *  # units
from pyoomph.expressions.phys_consts import gas_constant  # and the gas constant


def parse_args():
    parser = argparse.ArgumentParser(
        description="Droplet coalescence simulation with surfactants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Surfactant parameters
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Surfactant strength (β)")
    parser.add_argument("--Pe", type=float, default=1.0,
                        help="Péclet number")
    parser.add_argument("--Gamma0", type=float, default=0.8,
                        help="Initial surfactant concentration")
    # Geometry parameters
    parser.add_argument("--theta", type=float, default=20.0,
                        help="Contact angle in degrees")
    parser.add_argument("--hp", type=float, default=1e-4,
                        help="Precursor film thickness")
    parser.add_argument("--Lx", type=float, default=6.0,
                        help="Domain size")
    parser.add_argument("--N", type=int, default=1000,
                        help="Number of mesh elements")
    parser.add_argument("--max-refinement-level", type=int, default=6,
                        help="Max adaptive mesh refinement level")
    return parser.parse_args()


def print_parameters(args):
    """Print all simulation parameters before running."""
    print("\n" + "="*60)
    print("DROPLET COALESCENCE SIMULATION - PARAMETERS")
    print("="*60)
    print("\nSurfactant Parameters:")
    print(f"  beta (β)           = {args.beta}")
    print(f"  Pe (Péclet)        = {args.Pe}")
    print(f"  Gamma0 (Γ₀)        = {args.Gamma0}")
    print("\nGeometry Parameters:")
    print(f"  theta              = {args.theta}°")
    print(f"  hp (precursor)     = {args.hp}")
    print(f"  Lx (domain size)   = {args.Lx}")
    print(f"  N (elements)       = {args.N}")
    print(f"  max_refinement     = {args.max_refinement_level}")
    print("="*60 + "\n")

class PlotterTry(MatplotlibPlotter):
	def __init__(self, problem: Problem, filetrunk: str = "plot_{:05d}", fileext: str | List[str] = "png", eigenvector: int | None = None, eigenmode: MeshDataEigenModes = "abs", add_eigen_to_mesh_positions: bool = True, position_eigen_scale: float = 1):
		super().__init__(problem, filetrunk, fileext, eigenvector, eigenmode, add_eigen_to_mesh_positions, position_eigen_scale)

	def define_plot(self):
		p = self.get_problem()
		colorbar_1 = self.add_colorbar("h", cmap ='seismic', position = 'top center')

		self.set_view(-3, 0, 0 , 2.05)

		self.add_plot("domain/h", colorbar=colorbar_1)

class DropletCoalescence(Problem):
	def __init__(self, args):
		super(DropletCoalescence, self).__init__()
		# Geometry (see paper §2.1)
		self.L = 1                      # contact line radius (length scale)
		self.theta = args.theta * pi / 180  # contact angle (convert from degrees)
		self.R = self.L / sin(self.theta)   # sphere radius
		self.H = self.R - self.L / tan(self.theta)  # apex height
		self.hp = args.hp                   # precursor film thickness h_∞/L
		self.Lx = args.Lx                   # domain size [-Lx/2, Lx/2]
		self.N = args.N                     # number of elements
		self.max_refinement_level = args.max_refinement_level

		# Surfactant parameters (see paper §2.2)
		self.beta = args.beta       # surfactant strength (β)
		self.Pe = args.Pe           # Péclet number
		self.Gamma0 = args.Gamma0   # initial surfactant concentration
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
		Gamma_init = self.Gamma0 * (0.5 - 0.5*tanh(var("coordinate_x") / self.hp))
		
		eqs+=InitialCondition(Gamma=Gamma_init)
		eqs+=InitialCondition(h=h_init) 
		
		eqs+=SpatialErrorEstimator(h=1) # refine based on the height field
		eqs += TextFileOutput()

		# self+=TextFileOutputAlongLine(filename='profile', start=(-3,0), end=(3,0), N =200) @ "domain"
		
		self.add_equations(eqs@"domain") # adding the equation
		
if __name__=="__main__":
	args = parse_args()
	print_parameters(args)
	with DropletCoalescence(args) as problem:
		problem.run(100,outstep=0.1,startstep=0.01,maxstep=50,temporal_error=1,spatial_adapt=1)
