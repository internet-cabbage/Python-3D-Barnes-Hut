import numpy as np
import colour
import vispy.scene
from vispy.scene import visuals
import sys
from tqdm import tqdm

AntiSingularity = 1e9
Efficiency = 1.0
G = 6.67e-11
widthVal = 500
heightVal = 500


class tree():
    def __init__(self, bodies, xmin:float, xmax:float, ymin:float, ymax:float, zmin:float, zmax:float):
        # bodies is the list of all bodies that is being inserted
        self.bodies = bodies
        

        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = xmin, xmax, ymin, ymax, zmin, zmax
        self.root = node(self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax)

        # Insert the bodies
        for b in self.bodies:
            insertFine = self.root.insert(b)
            if not insertFine:
                raise RuntimeError('Failed body position:', b.pos, 'bounds:', self.xmin,  self.xmax,  self.ymin,  self.ymax, self.zmin, self.zmax)
    def calculateForceOnNode(self, target: body):
        return self.calculateForce(self.root, target)
    
    def calculateForce(self,node: node, target: body):

        # If the node is empty, then it exerts no force
        if node.tMass == 0:
            return np.array([0.0,0.0,0.0])
        # If the node contains only the node being considered
        if node.body is target and all(child is None for child in node.children):
            return np.array([0.0,0.0,0.0])
        # otherwise calculate the vector distance between the center of mass of the node and the position of the body
        dx = node.cx - target.pos[0]
        dy = node.cy - target.pos[1]
        dz = node.cz - target.pos[2]
        r2 = dx**2 + dy**2 + dz**2
        r = np.sqrt(r2 + AntiSingularity**2) # AntiSingularity smooths out some consequences of the approximation

        # Decide whether to use the approximation or not?
        # if s/r is less than theta it uses the approxmation
        # if the node has no child nodes, it uses the approxination (not really an approximation then tbh)
        s = node.length()
        if (s/r) < Efficiency or all(child is None for child in node.children):
            # NEWTONS LAW OF GRAVITATION!!!!!!!!!!!!!!!!
            fmag = G * target.mass * node.tMass / (r2 + AntiSingularity**2)
            fx = fmag * (dx/r)
            fy = fmag * (dy/r)
            fz = fmag * (dz/r)
            return np.array([fx,fy,fz])
        else:
            # If the nodes are too close to apply the approximation, it recursively enters its children
            # and sums the contributions of force from all of them
            totalforce = np.array([0.0,0.0,0.0])
            for child in node.children:
                if child is not None and child.tMass >0:
                    totalforce = totalforce + self.calculateForce(child,target)
            return totalforce





# Celestial body class
class body:
    def __init__(self, mass: float, x: float,vx: float,y: float,vy: float, z: float, vz: float):
        self.mass = mass
        self.x,self.vx,self.y,self.vy,self.z,self.vz = x,vx,y,vy,z,vz
        self.pos = np.array([self.x,self.y,self.z], dtype=float)
        self.vel = np.array([self.vx, self.vy, self.vz], dtype=float)
        self.R = self.mass ** (1/3)


class node():
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float):
        # The bounding box coordinates
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = xmin, xmax, ymin, ymax, zmin, zmax
        
        # Total mass of the node
        self.tMass = 0.0
        self.cx, self.cy, self.cz = 0.0,0.0,0.0

        # If the node is a leaf node, this variable stores the body data
        self.body = None
        self.children = [None,None,None,None,None,None,None,None] # Top left, top right, bottom left, bottom right
    
    def containsPoint(self, x,y,z):
        if (self.xmin <= x < self.xmax) and (self.ymin <= y < self.ymax) and (self.zmin <= z < self.zmax):
            return True
        else:
            return False

    def length(self):
        return (self.xmax - self.xmin)


    def hasParticle(self,b :body):
        return self.containsPoint(b.pos[0],b.pos[1], b.pos[2])
    
    def octDivide(self):
        xmid = (self.xmin + self.xmax) / 2.0
        ymid = (self.ymin + self.ymax) / 2.0
        zmid = (self.zmin + self.zmax) / 2.0
        # X (right, left)
        # Y (front, back)
        # Z (top, bottom)
        # Top quadrants
        self.children[0] = node(self.xmin, xmid, ymid, self.ymax, zmid, self.zmax) # Top front left
        self.children[1] = node(xmid, self.xmax, ymid, self.ymax, zmid, self.zmax) # Top front right
        self.children[2] = node(self.xmin, xmid, self.ymin, ymid, zmid, self.zmax) # Top back left
        self.children[3] = node(xmid, self.xmax, self.ymin, ymid, zmid, self.zmax) # Top back right
        # Bottom quadrants
        self.children[4] = node(self.xmin, xmid, ymid, self.ymax, self.zmin, zmid) # Bottom front left
        self.children[5] = node(xmid, self.xmax, ymid, self.ymax, self.zmin, zmid) # Bottom front right
        self.children[6] = node(self.xmin, xmid, self.ymin, ymid, self.zmin, zmid) # Bottom back left
        self.children[7] = node(xmid, self.xmax, self.ymin, ymid, self.zmin, zmid) # Bottom back right
    
    def deportIndex(self,x,y,z):
        # This calculates which child node to deport the node into
        xmid = (self.xmin + self.xmax) / 2.0
        ymid = (self.ymin + self.ymax) / 2.0
        zmid = (self.zmin + self.zmax) / 2.0
        top = int(z >= zmid)
        front = int(y > ymid)
        left = int(x < xmid)
        temp = (top, front, left)
        match temp:
            case (1,1,1):
                return 0
            case (1,1,0):
                return 1
            case (1,0,1):
                return 2
            case (1,0,0):
                return 3
            case (0,1,1):
                return 4
            case (0,1,0):
                return 5
            case (0,0,1):
                return 6
            case (0,0,0):
                return 7
        
    def insert(self, b: body) -> bool: # Returns True if the body is inserted, False otherwise
        # Is the body even within the node we're trying to inset it into?
        
        # Early on in my code I tried to pass a list of bodies as the argument of the self.insert() method
        # so this code just checks that I am not doing it again
        if not isinstance(b,body):
            raise TypeError('STOP INSERTING A LIST OF BODIES AS THE FUCKING ARGUMENT')
        
        if not self.hasParticle(b):
            return False

        # Case for if the node is empty:

        if self.tMass == 0.0:
            self.cx = b.pos[0]
            self.cy = b.pos[1]
            self.cz = b.pos[2]
            self.tMass = b.mass
            self.body = b
            return True # A great success!!!
        
        # For when node is an internal node with children, we shove it down another branch of the tree

        if any(child is not None for child in self.children):
            index = self.deportIndex(b.pos[0], b.pos[1], b.pos[2])
            if self.children[index] is None:
                # This menas that the child note has not been created, so it needs to be subdivided!!!
                self.octDivide()
            # This will keep trying to recursively insert the body
            insertedBody = self.children[index].insert(b)
            # This recomputes the center of mass, and its location
            totalMass = self.tMass + b.mass
            self.cx = (self.cx * self.tMass + b.pos[0] * b.mass) / totalMass
            self.cy = (self.cy * self.tMass + b.pos[1] * b.mass) / totalMass
            self.cz = (self.cz * self.tMass + b.pos[2] * b.mass) / totalMass
            self.tMass = totalMass
            return insertedBody
    
        # PENULTIMATE CASE!!! WOOOO!!!
        # This checks if the node is currently a leaf node, and it would therefore have to shove both
        # the node already present into a child node, and the node being inserted.
        
        if self.body is not None:
            # The original / native node
            nativeNode = self.body
            self.octDivide()
            nativeIndex = self.deportIndex(nativeNode.x, nativeNode.y, nativeNode.z)
            insertedNativeNode = self.children[nativeIndex].insert(nativeNode)
            self.body = None

            # Inserts the new node somewhere
            newIndex = self.deportIndex(b.pos[0], b.pos[1], b.pos[2])
            insertedBody = self.children[newIndex].insert(b)

            # This recomputes the center of mass, and its location
            totalMass = self.tMass + b.mass
            self.cx = (self.cx * self.tMass + b.pos[0] * b.mass) / totalMass
            self.cy = (self.cy * self.tMass + b.pos[1] * b.mass) / totalMass
            self.cz = (self.cz * self.tMass + b.pos[2] * b.mass) / totalMass
            self.tMass = totalMass
            return True




# ==============New simulation

class Simulation:
    def __init__(self, bodies, bounds, dt):
        self.bodies = bodies
        self.bounds = bounds
        self.dt = dt
        # Initiialise tree and forces
        self.tree = tree(self.bodies, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
        self.forces = [self.tree.calculateForceOnNode(b) for b in self.bodies]

    def step(self):
        dt = self.dt
        
        #Update positions
        for i,b in enumerate(self.bodies):
            accel = self.forces[i] / b.mass
            b.pos = b.pos + (b.vel * dt) + (0.5 * dt **2 * accel)
        
        # Rebuild tree from new positions

        self.tree = tree(self.bodies, self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3], self.bounds[4], self.bounds[5])

        # Update forces
        newForces = [self.tree.calculateForceOnNode(b) for b in self.bodies]

        # Euler step update
        for i,b in enumerate(self.bodies):
            b.vel = b.vel + 0.5 * (self.forces[i] + newForces[i]) / b.mass * dt
        
        self.forces = newForces

        # Return data in a format that vispy can handle

        return np.vstack([b.pos for b in self.bodies])


# ===================================== simulation
'''
def simulate(bodies,dt,nframes):
    # Initialise the tree and forces
    myTree = tree(bodies, bounds[0], bounds[1], bounds[2], bounds[3])
    forces = [myTree.calculateForceOnNode(b) for b in bodies]

    # Iterates the code loop

    for frame in range(nframes):
        # I just discovered what enumerate() is, its so sick!!!
        for i,b in enumerate(bodies):
            accel = forces[i] / b.mass
            # THIS IS SHITTY EULER STEP, CHANGE IT TO A LEAPFROG SOON!!!
            b.pos = b.pos + (b.vel * dt) + (0.5 * dt**2 * accel)
        myTree = tree(bodies, bounds[0], bounds[1], bounds[2], bounds[3])

        newforces = [myTree.calculateForceOnNode(b) for b in bodies]

        for i,b in enumerate(bodies):
            b.vel = b.vel + 0.5 * (forces[i] + newforces[i]) / b.mass * dt
        forces = newforces

    return bodies
'''



# Initial conditions and constants

# xmin, xmax, ymin, ymax
bounds = [-1e13,1e13,-1e13,1e13,-1e13,1e13]
num = 200

# Masses and sizes

masses = np.random.uniform(1.5e29, 2e30,num)

density = 1410
diameter = ((6*masses)/(density * np.pi))**0.3333
sizes = (diameter / 1.39e9) * 5

# ========================== Relate mass and colour

# For a main sequence star, L = M^3.5

luminosity0 = 3.828e26
mass0 = 1.989e30
sigma = 5.67e-8

luminosity = luminosity0 * (masses/mass0)**3.5

temp = (luminosity/(diameter**2 * sigma))**0.25
print(temp)

XY = colour.temperature.CCT_to_xy(temp)
XYZ = colour.xy_to_XYZ(XY)
RGB = colour.XYZ_to_sRGB(XYZ)
RGB = np.clip(RGB,0,1)
print(RGB)




#print(masses)

def randomBodies(num):
    pos = np.random.uniform(-1e11,1e11,[3,num])
    return pos

posVals = randomBodies(num)

bodies = [None]*num

for i in range(0,posVals.shape[1]):
    # mass, x,vx,y,vy
    bodies[i] = body(masses[i], posVals[0][i],0.0,posVals[1][i],0.0, posVals[2][i], 0.0)

dt = 200

sim = Simulation(bodies, bounds, dt)

# Display data using vispy

# Make the canvas and add a simple viewer
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()


# Initial plotting

initialPos = np.vstack([b.pos for b in bodies])
scatter = visuals.Markers(parent=view.scene)
scatter.set_data(initialPos, edge_width=0, face_color=(1,1,1,0.8), size=sizes)
view.add(scatter)
view.camera = 'turntable'
view.camera.set_range(x=(-1e9,1e9), y=(-1e9,1e9))
# axis
axis = visuals.XYZAxis(parent=view.scene)


# Create a scatter diagram and fill the data

stepCount = 8000
inc = 0
posData = np.zeros((stepCount+1,num,3))

for i in tqdm(range(0,stepCount)):
    pos = sim.step()
    posData[i] = pos

print('Finished computing bodies')

def frame(event):
    global inc
    if inc < stepCount:
        #print('Frame called')
        scatter.set_data(posData[inc], size=sizes, face_color=RGB)
        inc = inc + 1
    else:
        inc = 0

timer = vispy.app.Timer(0.03, connect=frame, start=True)

print('Finished rendering')




# Add an axis

code = np.random.randint(0,200000)
#np.savetxt(str(code),posData)

if sys.flags.interactive != 1:
    vispy.app.run()