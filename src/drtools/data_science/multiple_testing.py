""" 
This module was created to handle multiple testing 
methods. 

"""


class Bonferroni:
    """Method to control Family-Wise Error Rate (FWER)
    """
    pass


class HolmStepDown:
    """Holm's method, also known as Holm's step-down procedure or the Holm-Bonferroni 
    method, is an alternative to the Bonferroni procedure. Holm's method 
    controls the FWER, but it is less conservative than Bonferroni, in 
    the sense that it will reject more null hypotheses, typically resulting in fewer 
    Type II errors and hence greater power.
    """
    pass


class BenjaminiHochberg:
    """Procedure to control the False Doscpvery Rate (FDR)
    """
    pass


class PValueResampling:
    pass


class FDRResampling:
    pass