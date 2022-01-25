#----------------------------------------------------------------------------#
# Control 5
# Nombre: Gonzalo Cuevas
# Rut: 19.721.859-2
#----------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
import random

#----------------------------------------------------------------------------#
# Funciones
#----------------------------------------------------------------------------#
#Descripción: Realiza la modulación FSK de una tira binaria
#Entrada: arreglo. Representa la tira binaria
#         entero. Contiene la tasa de bits 
#Salida: arreglo. Representación de la modulación FSK.
def OOK(tiraBinaria, tasaBits):
    duracion = len(tiraBinaria)/tasaBits
    f0 = 75
    f1 = 125
    tiraAmpliada = []
    modulacion = []
    #Se amplia la tira binaria segun la tasa de Bits
    for i in range (0,len(tiraBinaria)):
        f=np.ones(tasaBits)
        x=f*tiraBinaria[i]
        tiraAmpliada = np.concatenate((tiraAmpliada,x))
    #Datos para modular
    tasa = len(tiraAmpliada)
    t = np.linspace(0,duracion,tasa)
    A = 0*np.cos(2*np.pi*f0*t)
    B = np.cos(2*np.pi*f1*t)
    
    #Se calcula la modulación segun el bit leído.
    for i in range(0,tasa):
        if tiraAmpliada[i]==0:
            mod0 = np.array([A[i]])
            modulacion=np.concatenate((modulacion,mod0))
        else:
            mod1 = np.array([B[i]])
            modulacion=np.concatenate((modulacion,mod1))
    return t,tiraAmpliada,modulacion

def demoduladorOOK(modulacion,tiempo,tasa):
    resultado = []
    bit = 0
    i = 0
    while i < len(modulacion):
        if modulacion[i] == 0:
            for j in range(0,tasa):
                bit = 0
                i=i+1
            resultado.append(bit)
        if i < len(modulacion):
            if modulacion[i] != 0:
                for j in range(0,tasa):
                    bit = 1
                    i=i+1
                resultado.append(bit)

    return np.array(resultado)
            
            
def ruido(senal, snr):
    ruido = np.random.normal(0, 1, len(senal))
    energia_s = np.sum(np.abs(senal) * np.abs(senal))
    energia_n = np.sum(np.abs(ruido) * np.abs(ruido))
    snr_lineal = np.exp(snr/10)
    sigma = np.sqrt(energia_s / (energia_n * snr_lineal))
    print('Desviacion ruido: ' + str(sigma))
    ruido = sigma * ruido
    awgn = senal + ruido
    return awgn


def error(bitsdemod, bits):
    contador = 0
    for i in range(len(bits)):
        if bits[i] != bitsdemod[i]:
            contador += 1
    print("Numero de errores en la transmision: ",contador)
    ber = float(contador / len(bits))
    print("Error rate: ",ber)
    return ber


def simulacionCanal( bitrate ):
    largosenal = 10000
    bits = np.random.randint(2, size = int(largosenal))
    colores = ['-b', '-g', '-r']
    plt.figure(1)
    for i in range(0, 3):
        snr_x = []
        ber_y = []
        bitrate = bitrate + 10
        tiempo, senal, modulacion = OOK(bits, bitrate)
        
        for snr in range(-2, 12, 1):
            awgn = ruido(modulacion, snr)
            demod = demoduladorOOK(awgn, tiempo, bitrate) 
            ber = error(demod, bits)
            snr_x.append(snr)
            ber_y.append(ber)
            lab = str(bitrate) + ' [bps]'
            plt.plot(snr_x, ber_y, colores[i], label=lab, marker="o")
   
    
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.xscale('linear')
    plt.title('Rendimiento SNR vs Bitrate')
    plt.legend()
    plt.show()


#----------------------------------------------------------------------------#
# Bloque principal
#----------------------------------------------------------------------------#

random.seed(102)    

#Tiras Binarias
tiraBinaria1 = random.choices([0,1], k=10)

#Modulación FSK
tasa = 10
tiempoOK,tiraOK,OOK1 = OOK(tiraBinaria1,10)

demodulacion = demoduladorOOK(OOK1,tiempoOK,tasa)

#Gráfico de la tira binaria de 10 bits
plt.subplot(2,1,1) 
plt.title("Tira binaria de 10 bits")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [db]")
plt.plot(tiempoOK,tiraOK,color ='chocolate')
plt.subplot(2,1,2) 
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [db]")
plt.plot(tiempoOK,OOK1,color ='orange')

plt.show()
##CANAL
simulacionCanal(tasa)   