# this is a part of Rigid Body simulation programm
# in this file we create class Point with several parametres
# first of all as was menshioned in rigidBody point have: mass, moment of intertia; linear and angular speed and also, friction coefficient

class InitError(Exception):
	def __init__(self, message):
		super().__init__(message)

class CenterOfMass:
	def __init__(self, body):
		...

class Point:

	def INITERROR(func):
		def _unwrapper(self, mass, position, velocity, friction):
			variableNames = {'mass'		: mass	  , 'position' : position,
							 'velocity' : velocity, 'friction' : friction}

			for variableName, variableValue, types in zip(  variableNames.keys(), variableNames.values(),
												[(int, float), (tuple), (tuple), (int, float)]):
				if not(isinstance(variableValue, types)):
					raise InitError(f"Instead of {variableValue.__class__}, {variableName} should be {types}")
				
				specialCases = ['position', 'velocity']

				if variableName == 'position' or variableName == 'velocity':
					if len(variableValue) > 2:
						raise InitError(f"Instead of {len(position)}, {variableName} should have length 2")
					for idx, internalValue in enumerate(variableValue):
						if not(isinstance(internalValue, (int, float))):
							raise InitError(f"Instead of {internalValue.__class__}, {variableName} should contain int | float on position {idx}")
			return func(self, mass, position, velocity, friction)
		return _unwrapper

	@INITERROR
	def __init__(self, 
				mass     :  int | float				  =  1    , # point's mass                ·🠒x                
				position : (int, int)				  = (0,0) , # point position according to ↓y   cartesian system (y,x)
				velocity : (int | float, int | float) = (1, 0), # velocity consist of 2 vectors: forward and angular velocity
				friction :  int | float				  =  0.5):  # point's friction coefficient, NOTE: in arbitrary system all point can have different coefficient
		self.mass = mass
		self.position = position
		self.velocity = velocity
		self.friction = friction
