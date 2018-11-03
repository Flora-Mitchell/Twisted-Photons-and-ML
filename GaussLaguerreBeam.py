import numpy
import matplotlib.pyplot as pyplot
import scipy.special.orthogonal

'''https://etd.ohiolink.edu/rws_etd/document/get/dayton1355513250/inline'''
lamda = 1550e-9
k = 2*numpy.pi/lamda
w0 = 0.003
z0 = (numpy.pi * w0**2)/lamda
z = 200
wz = w0*numpy.sqrt(1 + (z/z0)**2)
Rz = z * (1 + (z0/z)**2)
p = 0
l = 8
psi = (numpy.absolute(l) + 2*p + 1)*numpy.arctan(z/z0)

def Cart2Polar(x, y):
    r = numpy.sqrt(x**2 + y**2)
    phi = numpy.arctan2(y, x)
    return r, phi
    
def LaguerreGauss(r, phi):
    Alp = numpy.sqrt((2*scipy.special.factorial(p))/(numpy.pi*scipy.special.factorial(p+numpy.absolute(l))))
    LG = scipy.special.genlaguerre(p, l)
    rho = r/wz
    Im = Alp/wz * ((rho)*numpy.sqrt(2))**numpy.absolute(l) * (LG(2 * rho**2))
    Re = numpy.exp(-1* rho**2) * numpy.exp(-1j*k*(r**2/(2*Rz))) * numpy.exp(-1j*k*z) * numpy.exp(1j*psi)
    R = Im * Re
    return R * (numpy.exp(1j * l * phi) + numpy.exp(-1j * l * phi))
    
def IntensityProfile(x, y):
    r, phi = Cart2Polar(x, y)
    E = LaguerreGauss(r, phi)
    return (E * numpy.conj(E)).real     
                        
x = numpy.arange(-0.2, 0.2, 0.0008)
y = numpy.copy(x)
xx, yy = numpy.meshgrid(x, y)

Intensity = IntensityProfile(xx, yy)
pyplot.imshow(Intensity,cmap=pyplot.cm.gray)
pyplot.show()
