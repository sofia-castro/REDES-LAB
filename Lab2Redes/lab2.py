
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.fft import fft, ifft,fftfreq
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write
from numpy import pi, cos, sin, exp 
from scipy.integrate import simpson
from scipy import integrate
from scipy.signal import hilbert,firwin
import scipy.signal.signaltools as sigtool 



# Entradas: arreglo con el tiempo de la señal, arreglo con la señal, string con el titulo del gráfico, string con el color.
# Salida: No posee.
# Función: Realiza el gráfico de una señal en el dominio del tiempo.

def grafTiempo(tiempo,senal,titulo,color):
    plt.fill_between(tiempo, senal, color=color) 
    plt.xlim(tiempo[0], tiempo[-1])
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [db]')
    plt.grid(True)
    plt.title(titulo)
    plt.show()
    
# Entradas: arreglo con las frecuencias de la señal, arreglo con la señal, string con el titulo del gráfico, string con el color.
# Salida: No posee.
# Función: Realiza el gráfico de una señal en el dominio de la frecuencia.
def grafFrecuencia(frecuencias,serie,titulo,color):
    plt.plot(frecuencias, serie, color = color)
    plt.xlabel('W [Hz])')
    plt.ylabel('F(W)')
    plt.grid(True)
    plt.title(titulo)
    plt.show()

# Entrada: El tiempo (float) total de las señales, un arreglo de la señal original, un arreglo señal modulada , string con
# color a eleccción y otro string con color a elección.
# Salida: No posee.
# Función: Genera un gráfico comparativo entre la señal orginal y la señal modulada, con el fin de visualizar las diferencias.
def grafComparativo(tiempo_suma,signal_original,signal_modulada,color1,color2):
    plt.plot(tiempo_suma, signal_original, 'b-', label='Señal modulada', color = color1)
    plt.plot(tiempo_suma, signal_modulada, 'g-', linewidth=2, label='Señal original',color = color2)
    plt.xlabel('Tiempo')
    plt.grid()
    plt.legend()
    plt.show()
    
# Entrada: Arreglo de la señal, entero con la frecuencia de la señal y un float para el índice de modulación.
# Salida: un arreglo de la señal modulada en FM.
# Función: Consiste en realizar la modulación Fm para una señal, mediante la aplicación de su ecuación.   
def filtroPasoBajo(freq_muestra,senal,orden,freq_corte):
    nyq = 0.5 * freq_muestra
    corte_normalizado = freq_corte / nyq
    b, a = butter(orden, corte_normalizado, btype='low', analog=False)
    signal_filtrada = lfilter(b, a, senal)
    return signal_filtrada  
# Entrada: Arreglo de la señal, entero con la frecuencia de la señal y un float para el índice de modulación.
# Salida: un arreglo de la señal modulada en FM.
# Función: Consiste en realizar la modulación Fm para una señal, mediante la aplicación de su ecuación.
def FM(signal,frec,indice):
    tiempo =  len(signal)/frec
    x = np.linspace(0, tiempo, len(signal))   
    frecp = 10999                   
    integral= integrate.cumtrapz(signal,x, initial=0)
    modulacion_FM = np.cos( 2*frecp*np.pi*x + indice * integral)
    return modulacion_FM
    
# Entradas: arreglo con la señal, entero con la frecuencia de la señal, float para el índice de modulación.
# Salida: arreglo con la señal modulada en AM.
# Función: Realiza la modulación AM de una señal.
def AM(signal,frec,indice):
    tiempo =  len(signal)/frec
    x = np.linspace(0, tiempo, len(signal))  
    frecp=20000 
    portadora = cos(2*pi*frecp*x)
    modulacion_AM = indice*signal*portadora
    return modulacion_AM

# Entradas: arreglo con la señal modulada en AM, float del tiempo de la señal, entero para la frecuencia de la señal.
# Salida: arreglo con la señal demodulada en AM.
# Función: Realiza la demodulación AM de una señal modulada en AM.                                        
def AM_demodulacion(signal_modulada, time, fp):
    portadora = cos(2 * pi * fp* time)
    demodulada = signal_modulada * portadora 

    return filtroPasoBajo(fp,demodulada,6,3900)
    
# Entradas: arreglo con la señal modulada en FM.
# Salida: arreglo con la señal demodulada en FM.
# Función: Realiza la demodulación FM de una señal modulada en FM.
def FM_demodulacion(signal):
    #Derivada
    diff = np.diff(signal, 1) 
    y_env = np.abs(sigtool.hilbert(diff))
    y_env = list(y_env)
    y_env.insert(0,0)
    return np.array(y_env)*10000-15000

#Función para realizar el filtro de paso bajo.
  

#Obtención de la señal mensaje y sus respectivos datos.
dirHandel = "/Users/Acer/Documents/LabRedes/REDES-LAB/Lab2Redes/handel.wav"
fr_handel, signal_handel = wavfile.read(dirHandel)
tiempo_handel = np.arange(len(signal_handel))/float(fr_handel)

#Tiempo de la señal
tiempo =  len(signal_handel)/fr_handel
t = np.linspace(0, tiempo, len(signal_handel))  

#Signal original en el tiempo:
grafTiempo(tiempo_handel,signal_handel,"Señal Handel en el tiempo (original)","orange")

#Espectro Señal original
frec_H = fftfreq(len(signal_handel),1/fr_handel)
espectro_handel = fft(signal_handel)
grafFrecuencia(frec_H,espectro_handel,"Espectro Frecuencias señal original","orange")

#Ancho de banda de señal original

bandwidth_SO = np.max(signal_handel) - np.min(signal_handel)

#Índices de modulación
indice_modulacion1 = 1
indice_modulacion2 = 1.25

#MODULADOR AM--------------------------------------------------------------------------------

signal_modulada_1 = AM(signal_handel,fr_handel,indice_modulacion1)
signal_modulada_2 = AM(signal_handel,fr_handel,indice_modulacion2)


bandwidth_AM1 = np.max(signal_modulada_1) - np.min(signal_modulada_1)

bandwidth_AM2 = np.max(signal_modulada_2) - np.min(signal_modulada_2)

print("Ancho de banda para AM k=1\n")
print(bandwidth_AM1)
print("Ancho de banda para AM k=1.25\n")
print(bandwidth_AM2)

grafTiempo(t,signal_modulada_1,"Señal modulada AM1","red")
grafTiempo(t,signal_modulada_2,"Señal modulada AM2","red")

grafComparativo(t,signal_modulada_2,signal_handel,"orange","red")

#Espectro modulación AM 1
frec_AM = fftfreq(len(signal_modulada_1),1/fr_handel)
espectro_AM1 = fft(signal_modulada_1)

grafFrecuencia(frec_AM,espectro_AM1,"Espectro Frecuencias señal AM1","red")

#Espectro modulación AM 2
frec_AM = fftfreq(len(signal_modulada_2),1/fr_handel)
espectro_AM2 = fft(signal_modulada_2)
grafFrecuencia(frec_AM,espectro_AM2,"Espectro Frecuencias señal AM2","red")


#MODULADOR FM--------------------------------------------------------------------------------
s_fm_1 = FM(signal_handel,fr_handel,indice_modulacion1)
s_fm_2 = FM(signal_handel,fr_handel,indice_modulacion2)


bandwidth_FM1 = np.max(s_fm_1) - np.min(s_fm_1)
bandwidth_FM2 = np.max(s_fm_2) - np.min(s_fm_2)

print("Ancho de banda para FM k=1\n")
print(bandwidth_FM1)
print("Ancho de banda para FM k=1.25\n")
print(bandwidth_FM2)

grafTiempo(t,s_fm_1[:len(t)],"Señal modulada FM1","Blue")
grafTiempo(t,s_fm_2[:len(t)],"Señal modulada FM2","blue")

#Espectro modulación FM 1:
frec_FM1 = fftfreq(len(s_fm_1),1/fr_handel)
espectro_FM1 = fft(s_fm_1)
grafFrecuencia(frec_FM1,espectro_FM1,"Espectro Frecuencias señal FM1","blue")

#Espectro modulación FM 2:
frec_FM2 = fftfreq(len(s_fm_2),1/fr_handel)
espectro_FM2 = fft(s_fm_2)
grafFrecuencia(frec_FM2,espectro_FM2,"Espectro Frecuencias señal FM2","blue")

#DEMODULACIÓN---------------------------------------------------------------------------------

#Demodulacion AM:
demodulacion_AM1=AM_demodulacion(signal_modulada_1, t, 20000)
demodulacion_AM2=AM_demodulacion(signal_modulada_2, t, 20000)
grafTiempo(t,demodulacion_AM1,"Señal demodulada AM1","red")
grafTiempo(t,demodulacion_AM2,"Señal demodulada AM2","red")

#Demodulacion FM:
demodulacion_FM1= FM_demodulacion(s_fm_1)
demodulacion_FM2= FM_demodulacion(s_fm_2)

plt.plot(t,demodulacion_FM1,color = "blue")
plt.ylim(-22000, 22000)
plt.grid(True)
plt.title("Señal demodulada FM1")
plt.show()
plt.plot(t,demodulacion_FM2,color = "blue")
plt.ylim(-22000, 22000)
plt.grid(True)
plt.title("Señal demodulada FM2")
plt.show()


#https://medium.com/@nabanita.sarkar/simulating-amplitude-modulation-using-python-6ed03eb4e712
#https://www.youtube.com/watch?v=XEt47F3BV2k
#https://stackoverflow.com/questions/60193112/python-fm-demod-implementation

