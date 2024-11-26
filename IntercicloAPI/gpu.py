import pycuda.driver as cuda

try:
    cuda.init()
    device_count = cuda.Device.count()
    print(f"Dispositivos CUDA disponibles: {device_count}")
    if device_count > 0:
        for i in range(device_count):
            device = cuda.Device(i)
            print(f"Dispositivo {i}: {device.name()} con {device.total_memory() // (1024**2)} MB de memoria")
    else:
        print("No se detectó ningún dispositivo CUDA.")
except Exception as e:
    print(f"Error al inicializar CUDA: {e}")
