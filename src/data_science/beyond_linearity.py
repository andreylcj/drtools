""" 
This module was created to handle non-linear
models such as Polynomial Regression, Regression 
Splines, Smoothing Splines, Local Regression 
and Generalized Additive Models (GAMs).

"""


class PolynomialRegression:
    pass

    
class RegressionSpline:
    
    def spline():
        pass
    
    def natural_spline():
        pass
    
    def choosing_the_number_and_locations_of_the_knots(self):
        pass
    
    
class SmoothingSpline:
    
    def choosing_the_smoothing_parameter():
        """The leave-one-out cross-validation error (LOOCV) 
        can be computed very eﬃciently for smoothing splines, 
        with essentially the same cost as computing a single fit.
        """
        pass
    

class LocalRegression:
    pass


class GeneralizedAdditiveModel:
    pass


class RegressionGAM(GeneralizedAdditiveModel):
    """Regression by Generalized Additive Model.
    """
    pass


class ClassificationGAM(GeneralizedAdditiveModel):    
    """Classification by Generalized Additive Model.
    """
    pass
