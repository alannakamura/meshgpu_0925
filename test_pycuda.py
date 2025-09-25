import pycuda.driver as cuda
cuda.init()
print(f"{cuda.Device.count()} device(s) found.")
