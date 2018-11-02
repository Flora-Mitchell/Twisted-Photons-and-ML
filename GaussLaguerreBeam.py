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
l = 4
psi = (numpy.absolute(l) + 2*p + 1)*numpy.arctan(z/z0)


def LaguerreGauss(x, y):
    
    Alp = numpy.sqrt((2*scipy.special.factorial(p))/(numpy.pi*scipy.special.factorial(p+numpy.absolute(l))))
    LG = scipy.special.genlaguerre(p, l)
    
    r = numpy.sqrt(x**2 + y**2)
    phi = numpy.arctan2(y, x)
    rho = r/wz
    
    p1 = Alp/wz
    p2 = ((rho)*numpy.sqrt(2))**numpy.absolute(l)
    p3 = (LG(2 * rho**2))
    p4 = numpy.exp(-1* rho**2)
    p5 = numpy.exp(-1j*k*(r**2/(2*Rz)))
    p7 = numpy.exp(-1j*k*z)
    p8 = numpy.exp(1j*psi)
    
    R = p1 * p2 * p3 * p4 * p5 * p7 * p8
    #return R * numpy.conj(R) * 2 * (1 + numpy.cos(2 * l * phi))
    return R * (numpy.exp(1j * l * phi) + numpy.exp(-1j * l * phi))
        
x = numpy.arange(-0.2, 0.2, 0.0008)
y = numpy.copy(x)
xx, yy = numpy.meshgrid(x, y)



Z = LaguerreGauss(xx, yy)

Intensity = (Z * numpy.conj(Z)).real
pyplot.imshow(Intensity,cmap=pyplot.cm.gray)
pyplot.show()
