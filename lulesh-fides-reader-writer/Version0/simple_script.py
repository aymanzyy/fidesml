# Specify the Catalyst channel name
catalystChannel = "grid"

from paraview.simple import *
from paraview import catalyst
# Pipeline
print("CREATE ")

data = TrivialProducer(registrationName=catalystChannel)
extractor = CreateExtractor('VTPD', data)
# Catalyst options
options = catalyst.Options()
options.ExtractsOutputDirectory = "/home/local/KHQ/ayman.yousef/Downloads/ayman_internship_work/ayman_internship_work/lulesh-fides-reader-writer/datasets/"
options.GlobalTrigger.Frequency = 1
