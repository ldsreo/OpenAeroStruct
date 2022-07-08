import numpy as np

import openmdao.api as om


class TubeInsideWing(om.ExplicitComponent):
    """
    Create a constraint so the spar does not intersect the wing.
    Basically, the diameter must be less than or equal to the wing thickness.

    parameters
    ----------
    thickness[ny-1] : numpy array
        Thickness of each element of the FEM spar.
    radius[ny-1] : numpy array
        Radius of each element of the FEM spar.

    Returns
    -------
    thickness_intersects[ny-1] : numpy array
        If all the values are negative, each element does not intersect itself.
        If a value is positive, then the thickness within the hollow spar
        intersects itself and presents an impossible design.
        Add a constraint as
        `OASProblem.add_constraint('thickness_intersects', upper=0.)`
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        self.surface = surface = self.options["surface"]

        self.ny = surface["mesh"].shape[1]

        self.add_input("mesh", val=surface["mesh"], units="m")
        self.add_input("t_over_c", val=np.zeros(self.ny - 1))
        self.add_input("radius", val=np.zeros((self.ny - 1)), units="m")
        self.add_output("tube_in_wing", val=np.zeros((self.ny - 1)), units="m")

        c = np.zeros(self.ny)        

        arange = np.arange(self.ny)
        arange1 = np.arange(self.ny - 1)
        self.declare_partials("tube_in_wing", "radius", rows=arange1, cols=arange1, val=1.0)
        # self.declare_partials("tube_in_wing", "chord", rows=arange, cols=arange, val=input)

    def compute(self, inputs, outputs):
        c = np.sqrt(np.square(inputs["mesh"][0,:,0]-inputs["mesh"][1,:,0])+np.square(inputs["mesh"][0,:,2]-inputs["mesh"][1,:,2]))
        outputs["tube_in_wing"] = 2*inputs["radius"] - np.multiply(inputs["t_over_c"],.5*(c[:-1]+c[1:]))
