import sys,os
dir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir)
from PerceptualLoss import LossNetwork as PerLoss
from DenseDepthNet import NormDepth
from transfomer_backbone import TANet