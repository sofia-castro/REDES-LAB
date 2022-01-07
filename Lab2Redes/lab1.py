import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft,fftfreq
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write

#Función para graficar en el dominio del tiempo.
def grafTiempo(tiempo,senal,titulo,color):
    plt.fill_between(tiempo, senal, color=color) 
    plt.xlim(tiempo[0], tiempo[-1])
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title(titulo)
    plt.show()
    
#Función para graficar en el dominio de la frecuencia.
def grafFrecuencia(frecuencias,serie,titulo,color):
    plt.plot(frecuencias, np.abs(serie), color = color)
    plt.xlabel('Frecuencia')
    plt.ylabel('Poder')
    plt.title(titulo)
    plt.show()

#Función para graficar el espectrograma.
def espectrograma(freq,senal,titulo,color):
    plt.subplot(111)
    plt.specgram(senal,Fs=freq,cmap=color)
    plt.xlabel('Tiempo')
    plt.ylabel('Frecuencia')
    plt.title(titulo)
    plt.show()

#Función para realizar el filtro de paso bajo.
def filtroPasoBajo(freq_muestra,senal,orden,freq_corte):
    nyq = 0.5 * freq_muestra
    corte_normalizado = freq_corte / nyq
    b, a = butter(orden, corte_normalizado, btype='low', analog=False)
    signal_filtrada = lfilter(b, a, senal)
    return signal_filtrada

#Función para graficar la comparación de dos señales.
def grafComparativo(tiempo_suma,signal_ruidosa,signal_filtro,color1,color2):
    plt.plot(tiempo_suma, signal_ruidosa, 'b-', label='Señal Ruidosa', color = color1)
    plt.plot(tiempo_suma, signal_filtro, 'g-', linewidth=2, label='Señal filtrada',color = color2)
    plt.xlabel('Tiempo')
    plt.grid()
    plt.legend()
    plt.show()
    
#AUDIOS------------------------------------------------------------------------------------------------   
direccionAudioGonzalo = "/Users/Acer/Documents/LabRedes/Lab1Redes/audioGonzalo.wav"
direccionAudioSofia = "/Users/Acer/Documents/LabRedes/Lab1Redes/audioSofia.wav"
direccionRuidoRosa = "/Users/Acer/Documents/LabRedes/Lab1Redes/pink_noise.wav"

#AUDIO 1: Gonzalo Cuevas
frecuencia_gonza, signal_gonza = wavfile.read(direccionAudioGonzalo)
tiempo_gonza = np.arange(len(signal_gonza))/float(frecuencia_gonza)
tam_signalG = len(signal_gonza)
frecuencias_signalG = fftfreq(tam_signalG, 1 / frecuencia_gonza)

#AUDIO 2: Sofía Castro
frecuencia_sofia, signal_sofia = wavfile.read(direccionAudioSofia)
tiempo_sofia = np.arange(len(signal_sofia))/float(frecuencia_sofia)
tam_signalS = len(signal_sofia)
frecuencias_signalS = fftfreq(tam_signalS, 1 / frecuencia_sofia)

#Señales en el tiempo
grafTiempo(tiempo_gonza, signal_gonza, "Señal Gonzalo Cuevas", "orange")
grafTiempo(tiempo_sofia, signal_sofia, "Señal Sofía Castro", "mediumvioletred")

#Transformadas de Fourier
transformada_signalG = fft(signal_gonza)
transformada_signalS = fft(signal_sofia)
grafFrecuencia(frecuencias_signalG,transformada_signalG,"Transformada de Fourier señal Gonzalo","orange")
grafFrecuencia(frecuencias_signalS,transformada_signalS,"Transformada de Fourier señal Sofía","mediumvioletred")

#Inversas de las transformadas
inv_signalG = ifft(transformada_signalG)
inv_signalS = ifft(transformada_signalS)
grafTiempo(tiempo_gonza, inv_signalG, "Inversa de la serie Gonzalo", "orange")
grafTiempo(tiempo_sofia, inv_signalS, "Inversa de la serie Sofía", "mediumvioletred")

#Espectrogramas 
espectrograma(frecuencia_gonza,signal_gonza,"Espectrograma Gonzalo","afmhot")
espectrograma(frecuencia_sofia,signal_sofia,"Espectrograma Sofía","inferno")


#RUIDO: Pink Noise------------------------------------------------------------------------------------------

frecuencia_pink, signal_pink = wavfile.read(direccionRuidoRosa)
tiempo_pink= np.arange(len(signal_pink))/float(frecuencia_pink)
tam_pink = len(signal_pink)
frecuencias_pink= fftfreq(tam_pink, 1 / frecuencia_pink)

#Señal en el tiempo
grafTiempo(tiempo_pink, signal_pink, "Señal Pink Noise", "hotpink")

#Transformada de Fourier
transformada_pink = fft(signal_pink)
grafFrecuencia(frecuencias_pink,transformada_pink,"Transformada de Fourier señal Pink Noise","hotpink")

#Inversa de la Transformada de Fourier
inv_pink = ifft(transformada_pink)
grafTiempo(tiempo_pink, inv_pink, "inversa de la serie Pink Noise", "hotpink")

#Espectrograma 
espectrograma(frecuencia_pink,signal_pink,"Espectrograma Pink Noise","RdPu_r")

#SEÑAL RUIDOSA----------------------------------------------------------------------------------------------
signal_suma = signal_pink + signal_gonza
frecuencia_suma = frecuencia_pink
tiempo_suma = tiempo_pink
tam_suma = len(signal_suma)
frecuencias_suma= fftfreq(tam_suma, 1 / frecuencia_suma)

#Señal en el tiempo
grafTiempo(tiempo_suma, signal_suma, "Señal ruidosa", "steelblue")

#Transformada de Fourier
transformada_suma = fft(signal_suma)
grafFrecuencia(frecuencias_pink,transformada_pink,"Transformada de Fourier señal ruidosa","steelblue")

#Inversa de la Transformada de Fourier
inv_suma = ifft(transformada_suma)
grafTiempo(tiempo_suma, inv_suma, "inversa de la serie ruidosa", "steelblue")

#Espectrograma 
espectrograma(frecuencia_suma,signal_suma,"Espectrograma señal ruidosa","Blues_r")

#SEÑAL FILTRADA---------------------------------------------------------------------------------------------
signal_filtro = filtroPasoBajo(frecuencia_pink,signal_suma,6,850)

#Señal en el tiempo
grafTiempo(tiempo_suma, signal_filtro, "Señal filtrada", "mediumturquoise")

#Transformada de Fourier
transformada_filtrada = fft(signal_filtro)
grafFrecuencia(frecuencias_suma,transformada_pink,"Transformada de Fourier señal filtrada","mediumturquoise")

#Espectrograma 
espectrograma(frecuencia_suma,signal_filtro,"Espectrograma señal filtrada","YlGnBu_r")

#Comparación señal ruidosa y filtrada
grafComparativo(tiempo_suma,signal_suma,signal_filtro,"steelblue","mediumturquoise")

#EXPORTACIÓN DE AUDIOS-------------------------------------------------------------------------------------
y2 = np.int16(signal_filtro)
write("audio_filtro.wav", frecuencia_gonza,y2)
write("audio_ruidoso.wav", frecuencia_gonza,signal_suma)





















