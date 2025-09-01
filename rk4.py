import matplotlib.pyplot as plt
import math

def force_comp(x:float,y:float,i:float)->float: #Componentes da forca
  return (-1)*4*(math.pi**2)*i/(x**2+y**2)**(1.5)

def rk4_kepler(x0:float,y0:float,vx0:float,vy0:float,pontos:int):

  x = [x0]
  y = [y0]
  vx = [vx0]
  vy = [vy0]
  h = 1/pontos
  iter = 0

  while iter < pontos:
    kx1 = vx[iter]*h
    ky1 = vy[iter]*h
    kvx1 = force_comp(x[iter],y[iter],x[iter])*h
    kvy1 = force_comp(x[iter],y[iter],y[iter])*h

    kx2 = (vx[iter]+kvx1/2)*h
    ky2 = (vy[iter]+kvy1/2)*h
    kvx2 = force_comp(x[iter]+kx1*(0.5),y[iter]+ky1*(0.5),x[iter]+kx1*(0.5))*h
    kvy2 = force_comp(x[iter]+kx1*(0.5),y[iter]+ky1*(0.5),y[iter]+ky1*(0.5))*h

    kx3 = (vx[iter]+kvx2/2)*h
    ky3 = (vy[iter]+kvy2/2)*h
    kvx3 = force_comp(x[iter]+kx2*(0.5),y[iter]+ky2*(0.5),x[iter]+kx2*(0.5))*h
    kvy3 = force_comp(x[iter]+kx2*(0.5),y[iter]+ky2*(0.5),y[iter]+ky2*(0.5))*h

    kx4 = (vx[iter]+kvx3)*h
    ky4 = (vy[iter]+kvy3)*h
    kvx4 = force_comp(x[iter] + kx3,y[iter] + ky3,x[iter] + kx3)*h
    kvy4 = force_comp(x[iter] + kx3,y[iter] + ky3,y[iter] + ky3)*h

    x.append(x[iter] + (1/6)*(kx1 + 2*kx2 + 2*kx3 + kx4))
    y.append(y[iter] + (1/6)*(ky1 + 2*ky2 + 2*ky3 + ky4))
    vx.append(vx[iter]+(1/6)*(kvx1 + 2*kvx2 + 2*kvx3 + kvx4))
    vy.append(vy[iter]+(1/6)*(kvy1 + 2*kvy2 + 2*kvy3 + kvy4))

    iter = iter + 1


  plt.plot(x,y,'bo')
  plt.show()

rk4_kepler(0.9832,0.0,0.0,6.3897,365)