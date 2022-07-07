import numpy as np

import openmdao.api as om


class MonotonicConstraint(om.ExplicitComponent):
    """
    Produce a constraint that is violated if a user-chosen measure on the
    wing does not decrease monotonically from the root to the tip.

    Parameters
    ----------
    var_name : string
        The variable to which the user would like to apply the monotonic constraint.

    Returns
    -------
    monotonic[ny-1] : numpy array
        Values are greater than 0 if the constraint is violated.

    """

    def initialize(self):
        self.options.declare("var_name", types=str)
        self.options.declare("surface", types=dict)
        self.options.declare("size", types=int)

    def setup(self):

        self.surface = surface = self.options["surface"]
        self.var_name = self.options["var_name"]
        self.con_name = "monotonic_" + self.var_name

        self.symmetry = surface["symmetry"]

        size_in = self.options["size"]
        size_out = size_in-1

        self.add_input(self.var_name, val=np.zeros(size_in))
        self.add_output(self.con_name, val=np.zeros(size_out))

        rows = np.arange(0, size_out)
        rows = np.vstack((rows, rows)).flatten(order="F")

        cols = np.arange(1, size_out)
        cols = np.vstack((cols, cols)).flatten(order="F")
        cols = np.insert(cols, 0, 0)
        cols = np.append(cols, size_out)

        sparse_val = np.ones_like(rows)
        sparse_val[1::2] = -1
        if not self.symmetry:
            if size_in % 2 == 0:
                sparse_val[size_out - 1 :] *= -1
            else:
                sparse_val[size_out :] *= -1

        self.declare_partials(self.con_name, self.var_name, rows=rows, cols=cols, val=sparse_val)

    def compute(self, inputs, outputs):
        # Compute the difference between adjacent variable values
        diff = inputs[self.var_name][:-1] - inputs[self.var_name][1:]
        if self.symmetry:
            outputs[self.con_name] = diff
        else:
            ny2 = (self.ny - 1) // 2
            outputs[self.con_name][:ny2] = diff[:ny2]
            outputs[self.con_name][ny2:] = -diff[ny2:]
