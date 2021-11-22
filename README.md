# Draft:

## Topic

Using Yahoo Finance and the code obtained from the Towards Data Science webpage (https://towardsdatascience.com/downloading-historical-stock-prices-in-python-93f85f059c1f) in order to scrape the information containing the stock data (Date, Open, High, Low, Close, Adj Close, Volume, Name), the topic that will be covered is to create a portfolio in order to suggest people or companies to invest. By working on the data, the target is to create a model that helps predict possible ups and downs in the market using the historical stock prices. The main reason to select this topic is that this data is real and will be of great use to polish skills and make something that will be useful for someone that wants to invert in the stock market, with this in mind we could have the chance to create a company that allows us to advise people or even companies about how to invest their money in the market with potential good results or even take a better decisions in our own investments.
Using a time series model and machine learning models, specially Support Vector Machine and Artificial Neural Networks, the data will be tested and trained to try to accomplish a good percent of accuracy. For this, the data that will be used are samples taken in regular time intervals, with pandas, numpy and matplotlib, the time behavior can be observed. Also, it is important to note that Keras will be used and, a deeper research on exponential smoothing will be made.

# Communication protocols

To communicate with the team members, the next platforms will be used: -Slack: There was a new workspace created in order to keep in touch with all the team members.

* Zoom: Using the time during the bootcamp and office hours in order to keep working on the project. Also, there is the possibility to use it outside the bootcamp class or office 
hours.
* Google Meet: In case there is a need to discuss any topic related to the challenge, and there is no way to access Zoom, this will be another way to contact each team member.

## Github

A Github was created with the name of Project_B (https://github.com/LennethNova/Project_B) and each team member has a branch with their respective name in order to control the way the data will be uploaded.

## Machine Learning Model

For this data, different models will be used to test the accuracy. As previously mentioned in the Topic section, SVM, ANN and other models will be used to train and test the data. Libraries such as pandas, numpy and matplotlib will also be considered in this point, but the analysis will not be exclusive to the use of those since some others will be needed in order to accomplish the project. The challenge is to get the best accuracy possible for the models and to make a good use of the code to get the expected results to make an accurate portfolio of the suggested investments.

## Database

The dataset, as mentioned previously in the Topic section has the next content:

      Stock symbol
      
      Date
      
      Open
      
      High
      
      Low
      
      Closing
      
      Adjusting Closing price
      
      Volume traded

![image](https://user-images.githubusercontent.com/86340630/139751767-e636da64-2826-46b3-9ec0-83ff34d40b72.png)

![image](https://user-images.githubusercontent.com/86340630/139751786-a6049d04-c678-4ce1-bc42-96ae5e3b9d0b.png)

![Captura de pantalla (984)](https://user-images.githubusercontent.com/86340630/139752046-8a3b2bcc-87ac-42cb-a7f8-7e525a261827.png)

# Dashboard

For now, the dashboard will not be shown since the data needs to be complete in order to start visualizing the results, but the idea is to create a Dashboard that we can use it with our potential clients to be able to advise them in an easier and more visual way in with we can communicate the findings of the models in an understandable way so they can make and informed decision.


# Long Short-Term Memory Network (LSTM)

LSTM es una RNN que es entrenada usando Backpropagation Through Time y supera el problema del gradiente que desaparece. En lugar de neuronas el LSTM tiene bloques de memoria que se conectan en capas. Un bloque tiene componentes que la hacen más "lista" que clásicas neurona y una memoria para secuencias recientes. Un bloque contiene puertas que administran el estado del bloque y su output. Una unidad opera a través de una secuencia de inputs y cada puerta dentro del bloque usa una función sigmoidal para controlar si son "disparadas" o no, haciendo el cambio en el bloque y si la información viaja o no.

Hay 3 tipos de puertas dentro de una unidad de memoria:

Puerta de Olvido: Condicionalmente decide qué información descartar
Puerta Input: "" qué valores del input actualizarán el estado de memoria
Puerta Output: "" que outpus se tendrán basados en el input y la memoria.
Cada puerta tiene pesos que se aprenden durante el proceso.

![image](https://user-images.githubusercontent.com/86340630/142929530-9052af20-0f27-4bf5-abc0-817daa05b821.png)

![image](https://user-images.githubusercontent.com/86340630/142930778-7002f588-992c-4d88-91e6-ece961976a48.png)

# Modelos Tradicionales de Series de Tiempo

Hay muchas formas de modelar una serie de tiempo para hacer predicciones. La mayoría de los modelos de series de tiempo tienen como objetivo incorporar la tendencia, la estacionalidad y los componentes restantes al mismo tiempo que abordan la autocorrelación y la estacionariedad incrustadas en la serie de tiempo. Por ejemplo, el modelo autorregresivo (AR) discutido en la sección anterior aborda la autocorrelación en la serie de tiempo.

# ARIMA
Si combinamos la estacionariedad con la autorregresión y un modelo de media móvil (discutido más adelante en esta sección), obtenemos un modelo ARIMA. ARIMA es un acrónimo de AutoRegressive Integrated Moving Average, y tiene los siguientes componentes:

AR (p)

Representa la autorregresión, es decir, la regresión de la serie de tiempo sobre sí misma, como se discutió en la sección anterior, con el supuesto de que los valores de la serie actual dependen de sus valores anteriores con algún rezago (o varios rezagos). El retraso máximo en el modelo se denomina p.

I(d)

Representa el orden de integración. Es simplemente el número de diferencias necesarias para que la serie sea estacionaria.

MA (q)

Representa media móvil. Sin entrar en detalles, modela el error de la serie temporal; de nuevo, el supuesto es que el error actual depende del anterior con cierto rezago, al que se hace referencia como q.

La ecuación de la media móvil se escribe como:

yt = c + εt + θ1εt – 1 + θ2εt – 2

donde, εt es ruido blanco. Nos referimos a esto como un modelo MA (q) de orden q.

Combinando todos los componentes, el modelo ARIMA completo se puede escribir como:

yt ′ = φ1yt′ – 1 + ⋯ + φpyt′ – p + θ1εt – 1 + ⋯ + θqεt – q + εt donde yt 'es la serie diferenciada (puede haber sido diferenciada más de una vez).

Los predictores del lado derecho incluyen tanto valores rezagados de yt 'como errores rezagados. A esto lo llamamos un modelo ARIMA (p, d, q), donde p es el orden de la parte autorregresiva, d es el grado de primera diferenciación involucrado y q es el orden de la parte de la media móvil.

Las mismas condiciones de estacionariedad e invertibilidad que se utilizan para los modelos autorregresivos y de media móvil también se aplican a un modelo ARIMA.

![image](https://user-images.githubusercontent.com/86340630/142930979-8fe0c429-6850-4f8c-938c-41c0a4733af7.png)

Autocorrelación y Estacionarios
Hay muchas situaciones en las que los elementos consecutivos de una serie de tiempo muestran una correlación. Es decir, el comportamiento de los puntos secuenciales de la serie se afecta entre sí de manera dependiente.

La autocorrelación es la similitud entre observaciones en función del desfase temporal entre ellas. Estas relaciones se pueden modelar utilizando un modelo de autoregresión. El término autoregresión indica que es una regresión de la variable contra sí misma.

En un modelo de autoregresión, pronosticamos la variable de interés utilizando una combinación lineal de valores pasados de la variable.

Por tanto, un modelo autorregresivo de orden p se puede escribir como

yt = c + φ1yt – 1 + φ2yt – 2 + .... φpyt – p + ε

donde εt es ruido blanco.

Un modelo autorregresivo es como una regresión múltiple pero con valores rezagados de yt como predictores. Nos referimos a esto como un modelo AR (p), un modelo autorregresivo de orden p. Los modelos autorregresivos son notablemente flexibles para manejar una amplia gama de patrones de series de tiempo diferentes.

Estacionario
Se dice que una serie de tiempo es estacionaria si sus propiedades estadísticas no cambian con el tiempo. Así, una serie de tiempo con tendencia o con estacionalidad no es estacionaria, ya que la tendencia y la estacionalidad afectarán el valor de la serie de tiempo en diferentes momentos. Por otro lado, una serie de ruido blanco es estacionaria, ya que no importa cuando la observe; debería verse similar en cualquier momento.

# No estacionarias

![image](https://user-images.githubusercontent.com/86340630/142931274-7de9d13d-54c6-41a4-8e54-be639198f72e.png)


En la primera gráfica, podemos ver claramente que la media varía (aumenta) con el tiempo, lo que resulta en una tendencia al alza. Por tanto, esta es una serie no estacionaria. Para que una serie se clasifique como estacionaria, no debe presentar una tendencia.

Pasando al segundo gráfico, ciertamente no vemos una tendencia en la serie, pero la varianza de la serie es una función del tiempo. Una serie estacionaria debe tener una varianza constante; por lo tanto, esta serie también es una serie no estacionaria.

En el tercer gráfico, la propagación se acerca a medida que aumenta el tiempo, lo que implica que la covarianza es una función del tiempo.

# Estacionaria

![image](https://user-images.githubusercontent.com/86340630/142931194-43381a9a-8158-47b3-86bd-e9defae29230.png)


En este caso, la media, la varianza y la covarianza son constantes en el tiempo. Así es como se ve una serie de tiempo estacionaria. Sería más fácil predecir valores futuros utilizando este cuarto gráfico. La mayoría de los modelos estadísticos requieren que la serie sea estacionaria para realizar predicciones precisas y efectivas.

Las dos razones principales detrás de la no estacionariedad de una serie de tiempo son la tendencia y la estacionalidad, como se muestra en la figura de 3. Para utilizar modelos de predicción de series de tiempo, generalmente convertimos cualquier serie no estacionaria en una serie estacionaria, lo que facilita el modelado, ya que las propiedades estadísticas no cambian con el tiempo.

Diferenciar
La diferenciación es uno de los métodos utilizados para hacer estacionaria una serie de tiempo. En este método, calculamos la diferencia de términos consecutivos en la serie.

La diferenciación se realiza típicamente para deshacerse de la media variable. Matemáticamente, la diferenciación se puede escribir como: yt ′ = yt - yt – 1 donde yt es el valor en un tiempo t.

Cuando la serie diferenciada es ruido blanco, la serie original se denomina serie no estacionaria de grado uno.

![image](https://user-images.githubusercontent.com/86340630/142931442-1c51ad49-188e-4ff6-a81f-c06c82ea5f08.png)






















