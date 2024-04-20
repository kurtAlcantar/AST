import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import csv
import random
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller

import streamlit as st


# Opciones del menú
opciones_menu = ['Anomalias', 'Patrones', 'Estadistica Descriptiva','Análisis de Estacionariedad de una Serie Temporal']
st.sidebar.title("Análisis de Series Temporales")
# Selección del usuario
seleccion = st.sidebar.selectbox('Selecciona un tema:', opciones_menu)

st.sidebar.subheader("Facultad de Ingenieria Mecanica y Electrica")
st.sidebar.subheader("Universidad de Colima")
st.sidebar.subheader("Nombre: Christopher Kurt Alcantar Contreras")
st.sidebar.subheader("Materia: Analisis de series temporales")
st.sidebar.subheader("Profesor: Dr. Walter Alexander Mata Lopez")

st.sidebar.image("images-removebg-preview.png", use_column_width=True)


# Código 1
if seleccion == 'Anomalias':
    # Título de la aplicación
    st.title('Análisis de Datos de Temperatura')
    st.subheader('En este análisis se generan datos de temperatura de sensores y se realizan diversas visualizaciones y análisis de los datos.')
    # Parámetros
    num_meses = 6
    dias_por_mes = 30
    horas_por_dia = 24
    num_sensores = 10
    anomalia_probabilidad = 0.05  # Probabilidad de tener una anomalía en cada lectura

    # Generar datos simulados
    np.random.seed(0)
    fechas = np.arange(0, num_meses * dias_por_mes * horas_por_dia)
    fechas_datetime = [datetime(2024, 1, 1) + timedelta(hours=int(h)) for h in fechas]  # Convertir horas a fechas
    temperaturas = np.round(np.random.normal(25, 2, (num_sensores, len(fechas))), 2)

    # Introducir anomalías
    anomalias = []
    for i in range(num_sensores):
        for j in range(len(fechas)):
            if np.random.rand() < anomalia_probabilidad:
                temperatura_anomalia = np.random.choice(['pico', 'alta', 'baja', 'fluctuacion'])
                if temperatura_anomalia == 'pico':
                    temperaturas[i, j] += np.random.uniform(5, 15)
                    anomalias.append((fechas_datetime[j], temperaturas[i, j]))
                elif temperatura_anomalia == 'alta':
                    temperaturas[i, j] += np.random.uniform(2, 5)
                    anomalias.append((fechas_datetime[j], temperaturas[i, j]))
                elif temperatura_anomalia == 'baja':
                    temperaturas[i, j] -= np.random.uniform(2, 5)
                    anomalias.append((fechas_datetime[j], temperaturas[i, j]))
                elif temperatura_anomalia == 'fluctuacion':
                    temperaturas[i, j] += np.random.uniform(-5, 5)
                    anomalias.append((fechas_datetime[j], temperaturas[i, j]))

    # Guardar datos en un archivo CSV
    with open('datos_temperatura_simulados.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Fecha', 'Sensor ID', 'Temperatura (°C)'])
        for i in range(num_sensores):
            for j in range(len(fechas)):
                writer.writerow([fechas_datetime[j], i+1, temperaturas[i, j]])

    # Leer datos de un archivo CSV
    df = pd.read_csv('datos_temperatura_simulados.csv')
    # Calcular la temperatura máxima de cada sensor
    maximos_por_sensor = np.max(temperaturas, axis=1)
    # Calcular la temperatura mínima de cada sensor
    minimos_por_sensor = np.min(temperaturas, axis=1)


    # Convertir la columna 'Fecha' a tipo datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Mostrar los primeros 5 registros
    st.subheader('Primeros 5 registros:')
    st.write(df.head())

    # Mostrar los últimos 5 registros
    st.subheader('Últimos 5 registros:')
    st.write(df.tail())

    # Mostrar un resumen de los datos
    st.subheader('Resumen de los datos:')
    st.write(df.describe())

    # Mostrar los datos de un sensor específico
    sensor_id = st.slider('Seleccione el ID del sensor:', 1, num_sensores)
    df_sensor = df[df['Sensor ID'] == sensor_id]
    st.subheader(f'Datos del Sensor {sensor_id}:')
    st.write(df_sensor)

    # Input de fecha de inicio 
    st.markdown("<h3>Seleccione la fecha de inicio:</h3>", unsafe_allow_html=True)
    fecha_inicio = st.date_input('Fecha de inicio', key="fecha_inicio")

    # Input de fecha de fin 
    st.markdown("<h3>Seleccione la fecha de fin:</h3>", unsafe_allow_html=True)
    fecha_fin = st.date_input('Fecha de fin', key="fecha_fin")

    # Convertir las fechas de entrada a tipo datetime
    fecha_inicio = datetime.combine(fecha_inicio, datetime.min.time())
    fecha_fin = datetime.combine(fecha_fin, datetime.min.time())

    # Filtrar el DataFrame con las fechas convertidas
    df_fecha = df[(df['Fecha'] >= fecha_inicio) & (df['Fecha'] < fecha_fin)]
    st.markdown("<h6>Nota: Seleccione rangos de fechas validos, de no ser validos, no se mostrar ningun registro.</h6>", unsafe_allow_html=True)
    # Mostrar los datos dentro del rango de fechas seleccionado
    st.subheader('Datos dentro del rango de fechas seleccionado:')
    st.write(df_fecha)

    # Mostrar los datos de un rango de temperaturas específico
    temp_min = st.slider('Seleccione la temperatura mínima:', min_value=0, max_value=40, value=20)
    temp_max = st.slider('Seleccione la temperatura máxima:', min_value=0, max_value=40, value=30)
    df_temp = df[(df['Temperatura (°C)'] >= temp_min) & (df['Temperatura (°C)'] < temp_max)]
    st.subheader('Datos dentro del rango de temperaturas seleccionado:')
    st.write(df_temp)

    # Calcular la media de cada sensor
    medias_por_sensor = np.mean(temperaturas, axis=1)

    # Gráfico de barras de la temperatura media de cada sensor
    st.subheader('Gráfico de barras de la temperatura media de cada sensor:')
    fig, ax = plt.subplots()
    ax.bar(np.arange(num_sensores), medias_por_sensor, color='skyblue')
    ax.set_title('Media de Temperatura de Cada Sensor')
    ax.set_xlabel('Sensor ID')
    ax.set_ylabel('Temperatura Media (°C)')
    ax.set_xticks(np.arange(num_sensores))
    ax.set_xticklabels([f'Sensor {i+1}' for i in range(num_sensores)])
    st.pyplot(fig)

    # Gráfico de la temperatura máxima de cada sensor
    st.subheader('Gráfico de la temperatura máxima de cada sensor:')
    fig, ax = plt.subplots()
    ax.bar(np.arange(num_sensores), maximos_por_sensor, color='salmon')
    ax.set_title('Temperatura Máxima de Cada Sensor')
    ax.set_xlabel('Sensor ID')
    ax.set_ylabel('Temperatura Máxima (°C)')
    ax.set_xticks(np.arange(num_sensores))
    ax.set_xticklabels([f'Sensor {i+1}' for i in range(num_sensores)])
    st.pyplot(fig)

    # Gráfico de la temperatura mínima de cada sensor
    st.subheader('Gráfico de la temperatura mínima de cada sensor:')
    fig, ax = plt.subplots()
    ax.bar(np.arange(num_sensores), minimos_por_sensor, color='lightgreen')
    ax.set_title('Temperatura Mínima de Cada Sensor')
    ax.set_xlabel('Sensor ID')
    ax.set_ylabel('Temperatura Mínima (°C)')
    ax.set_xticks(np.arange(num_sensores))
    ax.set_xticklabels([f'Sensor {i+1}' for i in range(num_sensores)])
    st.pyplot(fig)

    # Calcular la temperatura promedio de cada sensor por mes
    temperaturas_por_mes = temperaturas.reshape(num_sensores, num_meses, dias_por_mes, horas_por_dia)
    temperaturas_promedio_por_mes = np.mean(temperaturas_por_mes, axis=(2, 3))

    # Gráfico de barras de la temperatura promedio de cada sensor por mes
    st.subheader('Gráfico de barras de la temperatura promedio de cada sensor por mes:')
    fig, ax = plt.subplots()
    for i in range(num_sensores):
        ax.bar(np.arange(num_meses), temperaturas_promedio_por_mes[i], alpha=0.7, label=f'Sensor {i+1}')
    ax.set_title('Temperatura Promedio de Cada Sensor por Mes')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Temperatura Promedio (°C)')
    ax.set_xticks(np.arange(num_meses))
    ax.set_xticklabels([f'Mes {i+1}' for i in range(num_meses)])
    ax.legend()
    st.pyplot(fig)

    # Graficar lecturas de temperatura de un sensor para detectar anomalías
    st.markdown("<h3>Deslice para seleccionar entre los distintos sensores :</h3>", unsafe_allow_html=True)
    sensor_id = st.slider('', 1, num_sensores)
    st.subheader(f'Gráfico de lecturas de temperatura del Sensor {sensor_id}:')
    fig, ax = plt.subplots()
    ax.plot(fechas_datetime, temperaturas[sensor_id - 1], color='blue', alpha=0.7)
    ax.set_title(f'Lecturas de Temperatura del Sensor {sensor_id}')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Temperatura (°C)')
    st.pyplot(fig)

    # Conclusiones

    st.markdown("<h3>Conclusiones:</h3>", unsafe_allow_html=True)
    # Definir texto con saltos de línea usando triple comilla doble
    conclusiones = """
    - ¿Existen lecturas de temperatura que se desvíen significativamente del rango esperado para esa área de la planta?
    Si, existen lecturas de temperatura que son menores a las esperadas para esa área de la planta, lo que puede ser un indicio de que el sensor de temperatura no está funcionando correctamente, ya que el rango esperado para la temperatura es de 15 a 35 grados centígrados. Aun teniendo este rango hay lecturas de menos de 15 grados centígrados y mayores a 35 grados centígrados.

    - ¿Hay algún patrón o tendencia en las lecturas anómalas?
    Si, hay un patrón en las lecturas anómalas, ya que la mayoría de las lecturas anómalas se encuentran en el rango de 0 a 15 y de 35 a 50 grados centígrados.

    - ¿Qué características tienen las lecturas anómalas en comparación con las lecturas normales?
    Las lecturas anómalas tienen una temperatura menor a 15 grados centígrados y mayor a 35 grados centígrados, mientras que las lecturas normales se encuentran en el rango de 15 a 35 grados centígrados. Mayormente las temeperaturas normales tienen un rango mas proximo al 25 y 30 grados centígrados.

    - Existen lecturas de temperatura que se desvían significativamente del rango esperado para esa área de la planta.
    - Hay un patrón en las lecturas anómalas, ya que la mayoría de las lecturas anómalas se encuentran en el rango de 0 a 15 y de 35 a 50 grados centígrados
    """

    # Mostrar el texto en la aplicación
    st.write(conclusiones)
        
# Código 2
elif seleccion == 'Patrones':
    st.title('Análisis de Datos de Ventas')
    st.subheader('En este análisis se generan datos de ventas de productos electrónicos y se realizan diversas visualizaciones y análisis de los datos.')
    def generar_datos():
        productos = ['MacBook Pro', 'Airpods Pro', 'iPhone 12 Pro Max']
        ventas = []
        for producto in productos:
            for year in range(2018, 2021):
                for month in range(1, 13):
                    venta = {
                        'producto': producto,
                        'Año': year,
                        'Mes': month,
                        'Venta': random.randint(0, 300)
                    }
                    ventas.append(venta)
        return ventas

    
    def guardar_datos(ventas):
        with open('ventas.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['producto', 'Año', 'Mes', 'Venta'])
            writer.writeheader()
            for venta in ventas:
                writer.writerow(venta)

    # Programa principal
    if __name__ == "__main__":
        # Generar y guardar los datos
        ventas = generar_datos()
        guardar_datos(ventas)

        # Leer datos de un archivo CSV
        df = pd.read_csv('ventas.csv')

        # Mostrar los primeros 5 registros
        st.subheader('Primeros 5 registros:')
        st.write(df.head())

        # Mostrar los últimos 5 registros
        st.subheader('Últimos 5 registros:')
        st.write(df.tail())

        # Mostrar un resumen de los datos
        st.subheader('Resumen de los datos:')
        st.write(df.describe())

        # Filtrar datos por producto
        producto_seleccionado = st.selectbox('Seleccione un producto:', df['producto'].unique())
        st.write(df[df['producto'] == producto_seleccionado])

        # Graficar las ventas por producto
        ventas_por_producto = df.groupby('producto')['Venta'].sum()
        st.subheader('Ventas por producto:')
        st.bar_chart(ventas_por_producto)

        # Graficar las ventas por mes
        ventas_por_mes = df.groupby('Mes')['Venta'].sum()
        st.subheader('Ventas por mes:')
        st.bar_chart(ventas_por_mes)

        # Descomposición de la serie de tiempo para cada producto
        st.subheader('Descomposición de la serie de tiempo:')
        for producto in df['producto'].unique():
            df_producto = df[df['producto'] == producto]
            result = seasonal_decompose(df_producto['Venta'], model='additive', period=12)
            st.write(f'Descomposición de la serie de tiempo para {producto}:')
            st.line_chart(result.trend)
            st.line_chart(result.seasonal)
            st.line_chart(result.resid)
            st.line_chart(result.observed)
        st.subheader('Conclusiones:')
        conclusiones = """En el analisis de la tendencia estacional para cada 
        producto observamos que no hay una tendencia clara, por lo que no se 
        puede predecir el comportamiento de los datos. Sin embargo se puede observar 
        que los datos se comportan de manera similar en el añi 2018 y 2020 en los meses de enero.

¿Se observa alguna tendencia de crecimiento o decrecimiento en las ventas a lo largo del tiempo?
- En la totalidad de las ventas se observa varios crecimientos, por ejemplo; en la mackbook pro, en el año 2018 se observa un crecimiento en las ventas en el mes de enero, en el año 2019 se observa un crecimiento en las ventas en el mes de mayo y en el año 2020 se observa un crecimiento en las ventas en el mes de mayo y septiembre. 
- En los Airpods se observa un crecimiento en las ventas en el año 2018 en el mes de enero y en el año 2020 en el mes de enero. 
- En el iphone se observa un crecimiento en las ventas en el año 2018 en el mes de enero y en el año 2020 en el mes de enero y septiembre. En el iPhone 12 Pro Max en el mes de enero del 2018 en el mes de septiembre del 2019 y en el mes de septiembre del 2020

Como podemos observar en los datos, no hay una tendencia muy clara, sin embargo, en los meses de enero y septiembre se observa un crecimiento en las ventas de tres productos.
        """
        st.write(conclusiones)

# Análisis de Datos de Salarios
elif seleccion == 'Estadistica Descriptiva':
        st.title('Análisis de Datos de Salarios')
        st.subheader('En este análisis se generan datos de salarios anuales y se realizan diversas visualizaciones y análisis de los datos.')

        # Semilla aleatoria
        np.random.seed(97531)
        # Generar datos de salarios anuales (simulados)
        salarios = np.random.normal(loc=50000, scale=15000, size=250)
        salarios = np.round(salarios, -3)
        # Asegurarse de que todos los salarios sean positivos
        salarios = np.abs(salarios)

        # Mostrar los primeros 5 registros
        st.subheader('Primeros 5 registros:')
        st.write(salarios[:5])

        # Mostrar los últimos 5 registros
        st.subheader('Últimos 5 registros:')
        st.write(salarios[-5:])

        # Realizemos un análisis exploratorio de los datos
        media_salarios = np.mean(salarios)
        moda_salarios = stats.mode(salarios)[0][0]
        mediana_salarios = np.median(salarios)
        std_salarios = np.std(salarios)
        salario_minimo = np.min(salarios)
        salario_maximo = np.max(salarios)

        # Interpretación de los resultados de manera gráfica
        st.subheader('Histograma de Salarios')
        fig, ax = plt.subplots()
        ax.hist(salarios, bins=12, color='lightblue', edgecolor='black')
        st.pyplot(fig)

        # Gráfica de pastel de los salarios distribuidos en dos categorías (menos de 50000 y 50000 o más)
        st.subheader('Distribución de salarios')
        labels = ['Menos de 50000', '50000 o más']
        sizes = [np.sum(salarios < 50000), np.sum(salarios >= 50000)]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
        ax.set_title('Distribución de salarios')
        st.pyplot(fig)

        # Gráfica de caja
        st.subheader('Diagrama de Caja de Salarios')
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.boxplot(salarios, vert=False)
        ax.set_title('Distribución de salarios')
        st.pyplot(fig)

        # Gráfica de dispersión con media
        st.subheader('Gráfico de Dispersión de Salarios')
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(range(len(salarios)), salarios, color='blue', label='Salario anual')
        ax.axhline(media_salarios, color='red', linestyle='--', label='Media')
        ax.set_title('Salarios anuales')
        ax.set_ylabel('Salario anual')
        ax.legend()
        st.pyplot(fig)

        # Mostrar resultados numéricos
        st.subheader('Resultados Numéricos:')
        st.write(f"Media: {media_salarios}")
        st.write(f"Moda: {moda_salarios}")
        st.write(f"Mediana: {mediana_salarios}")
        st.write(f"Desviación Estándar: {std_salarios}")
        st.write(f"Salario Mínimo: {salario_minimo}")
        st.write(f"Salario Máximo: {salario_maximo}")

        # conclusiones
        st.subheader('Conclusiones:')
        conclusiones="""
En el desarrollo de esta actividad, diversos factores influyen en la obtención y la interpretación de los datos. Uno de los más destacados es la generación aleatoria de datos, la cual se fundamenta en una semilla de la cual se derivan los datos "aleatorios". Al generarlos de esta manera, podemos observar la similitud que existe entre los distintos conjuntos de datos.

¿Cómo llegamos a esta conclusión? Es bastante simple: basta con analizar las medidas de tendencia central. Si observamos detenidamente, los datos oscilan entre 10,000 y 80,000, y las medidas de tendencia central nos indican que los salarios tienen una media aritmética de 50,4076. Esto sugiere que nuestros datos son equitativos en términos subjetivos. Al representarlos en un gráfico de pastel, dividiendo los salarios en rangos a partir de la media (salarios menores a 50,000 y mayores a 50,000), podemos corroborar una vez más la equidad de los datos. Esto se refleja en el porcentaje de 52.4% y 47.6% respectivamente, lo que evidencia un equilibrio notable en la distribución de los salarios.

Además, es importante tener en cuenta que la generación aleatoria de datos no solo proporciona una muestra representativa de la población en estudio, sino que también permite reducir el sesgo y la influencia de variables externas que podrían afectar los resultados. Esto aumenta la confiabilidad de los datos y facilita su interpretación, contribuyendo así a una toma de decisiones más informada y precisa."""

        st.write(conclusiones)
elif seleccion == 'Análisis de Estacionariedad de una Serie Temporal':
    st.title('Análisis de Series Temporales')
    st.subheader('En este análisis se generan datos de una serie temporal y se realizan diversas visualizaciones y análisis de los datos.')
    # Generar datos de la serie temporal
    np.random.seed(0)
    t = np.arange(120)
    data = 20 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(size=120)
    serie_temporal = pd.Series(data, index=pd.date_range(start='2010-01-01', periods=120, freq='M'))

    # Redondear a dos decimales la temperatura
    serie_temporal = serie_temporal.round(2)

    # Guardar en un archivo csv
    serie_temporal.to_csv('serie_temporal.csv', header=False)

    # Cargar datos desde el archivo CSV
    serie_temporal = pd.read_csv('serie_temporal.csv', header=None, index_col=0)

    # Desplegar los primeros 5 registros
    st.subheader("Primeros 5 registros:")
    st.write(serie_temporal.head())

    # Desplegar los últimos 5 registros
    st.subheader("Últimos 5 registros:")
    st.write(serie_temporal.tail())

    # Desplegar la información general del archivo
    st.subheader("Información general:")
    st.write(serie_temporal.describe())

    # Imprimir la cantidad de registros
    st.write(f"Cantidad de registros: {len(serie_temporal)}")

    # Aplicar suavización exponencial simple
    serie_temporal_suavizada = serie_temporal.ewm(alpha=0.2).mean()

    # Graficar la serie original y la suavizada
    st.subheader("Serie Original y Suavizada:")
    fig, ax = plt.subplots()
    serie_temporal.plot(ax=ax)
    serie_temporal_suavizada.plot(ax=ax)
    ax.legend(['Original', 'Suavizada'])
    st.pyplot(fig)

    # Diferenciacion primera
    serie_temporal_diff = serie_temporal.diff()

    # Graficar la serie original y la diferenciada
    st.subheader("Serie Original y Diferenciada:")
    fig, ax = plt.subplots()
    serie_temporal.plot(ax=ax)
    serie_temporal_diff.plot(ax=ax)
    ax.legend(['Original', 'Diferenciada'])
    st.pyplot(fig)

    # Prueba de estacionariedad de la serie
    resultado_adf = adfuller(serie_temporal.squeeze())
    st.write("ADF Static:", resultado_adf[0])
    st.write("p-value:", resultado_adf[1])
    if resultado_adf[1] < 0.05:
        st.write("La serie es estacionaria.")
    else:
        st.write("La serie no es estacionaria.")

    # Graficar la serie temporal con tendencia lineal
    st.subheader("Serie de Tiempo con Tendencia Lineal:")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(serie_temporal, label='Serie de tiempo')
    ax.set_title('Serie de Tiempo con Tendencia Lineal')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Valor')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    #concluciones
    st.subheader('Conclusiones:')
    conclusiones = """Finalmente, mostraremos las conclusiones obtenidas a partir de los analisis realizados.
Como podemos observar despues de aplicar las distintas tecnicas y prueba, la serie no es estacionaria ya que si analizamos bien, la media no es constante y la varianza tampoco lo es. Podemos ver como en las graficas la media y la varianza cambian a lo largo del tiempo y no se mantienen constantes en ningun momento es por eso que vemos como si la grafica fuera creciente y decreciente en algunos momentos que no se mantienen constantes y estan variando a lo largo del tiempo."""

    st.write(conclusiones)
