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




























