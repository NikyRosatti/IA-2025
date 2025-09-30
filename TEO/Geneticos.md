# Algoritmos Geneticos

Los **algoritmos geneticos** son una familia de algoritmos de busqueda inspirados en los **principios de la evolucion** en la naturaleza. 

Imita proceso de seleccion natural y reproducion.

Alta calidad de soluciones para diverss problemas que involucran busqueda, optimizacion y aprendizaje

**Principio de variacion:** Rasgos y atributos de los individuos pueden variar

**Principio de herencia:** Rasgos se heredan de manera consistente a descendientes

**Principio de seleccion:**  Los individuos que poseen rasgos mejor adaptados al entorno tendrán 
más probabilidades de sobrevivir y tendran mas descendientes en proximas generaciones 

Son Algoritmos de **busqueda informada**, busqueda de **configuraciones exitosas** a partir de ciertas **configuraciones iniciales**, mediante la aplicacion de **reglas predifinidas de reconfiguracion o avance**

En los algoritmos genéticos, una **población de soluciones candidatas evoluciona de forma iterativa**: las mejores soluciones tienen más probabilidades de ser seleccionadas y transmitir sus características a la siguiente generación, logrando así **mejorar progresivamente la calidad de las soluciones al problema.**

## Genotipo

>El **genotipo** es una coleccion de genes agrupados en cromosomas

Cuando dos individuos se reproducen para crear descendencia, cada cromosoma de la descendencia contiene una combinacion de genes de ambos progenitores.

A los posibles valores que puede tomar un **Gen** se lo denomina **ALELO**.

## Poblacion 
La poblacion es un conjunto de soluciones candidatas para el problema (o coleccion de cromosomas).

## Evaluacion (Fitness)
Los individuos se **evalúan mediante una función de aptitud** (también llamada 
función objetivo), que es la función que buscamos optimizar o el problema que 
intentamos resolver.

Los  individuos  con  una  **mejor puntuación**  de  aptitud  representan  mejores soluciones y **tienen más probabilidades de ser seleccionados** para cruzarse (combinarse)  y  **formar parte de  la siguiente generación**. Con el tiempo(iteraciones), la calidad de las soluciones mejora, los valores de aptitud aumentan y **el proceso puede detenerse cuando se encuentra una solución con un valor de aptitud satisfactorio**

-La función de Fitness debe calificar a un 
individuo (solución candidata) en términos 
de del objetivo (solución)

Es deseable que la función sea lo menos 
ambigua posible, es decir, que permita 
distinguir a cada individuo de la manera 
más clara y precisa posible

## Evolucion
Para hacer evolucionar una población, se **eligen** individuos para realizar la **combinación** y/o la **mutación** el tamaño de la población.

## Seleccion
Se seleccionan aquellos con mayor puntuacion para combinarse y producir la siguiente generacion.

-Calificacion en base a la funcion de **Fitness**

-Seleccion para generar la proxima generacion

## Combinacion (crossover)
Para crear una nueva pareja de individuos,normalmente se eligen dos padres de la generación actual y se intercambian partes de sus cromosomas 
(combinación) para generar dos nuevos cromosomas  que representan a la **descendencia**. Esta operación se denomina **crossover**

Se debe establecer cómo es la configuración de combinación

## Mutacion 
Una **mutacion produce un cambio imprevisto** en algun individuo de la poblacion

La **mutacion ayuda a mantener la diversidad en la poblacion**, y en muchos casos a recuperar informacion perdida en la evolucion de la poblacion.

Una de las formas mas comunes de mutacion es el cambio en el valor de un gen de un cromosoma, por otro de sus alelos

## Eliminacion 
Es esencial mantener acotado el tamaño de las poblaciones 
Por esta razon, durante la evolucion se deben eliminar individuos de la poblacion. Los individuos elegidos para eliminar son aquellos "menos aptos" (de acuerdo a la funcioin de fitness)

Las características clave que distinguen a los algoritmos genéticos son: 
 
    • Mantener una población de soluciones.

    • Usar una representación genética de las soluciones. 
    
    • Emplear una función de aptitud para evaluar los resultados. 
    
    • Presentar un comportamiento probabilístico

## Ventajas
• Capacidad de optimización global. 
 • Manejo de problemas con representaciones matemáticas complejas.  
 • Resiliencia al ruido. 
 • Compatibilidad con el paralelismo y el procesamiento distribuido. 
 • Adecuación para el aprendizaje continuo

 ## Limitaciones
 • Necesidad de definiciones específicas del problema. 
 • Requieren ajuste de hiperparámetros. 
 • Operaciones computacionalmente intensivas. 
 • Riesgo de convergencia prematura. 
 • No garantizan encontrar una solución óptima

 ## Estructura
**Inicializar la poblacion**: se debe definir la configuracion de in individuo y sus alelos. La poblacion general se crea de manera aleatoria
**Evaluar la poblacion:** Se debe establecer la funcion de Fitness para ser aplicada a la poblacion a evaluar
Seleccion de individuos: Se deben establecer los hiperparametros que regula la cantidad
Crossover y Mutacion: Se debem establecer los hiperparametros como la configuracion de crossover y mutaciones. Ademas la probabilidad de ocurrencia de las mismas.
Evaluar Poblacion
Descartar individuos: Se deben establecer los hiperparametros que regula la cantidad
Evaluar condicion de terminacion: Se establece el criterio de terminacion
Retornar mejor individuo

### Seleccion
Seleccion por ruleta: cada individuo tiene una probabilidad de ser elegido proporcional a su aptitud, como si ocupara una porción de una ruleta cuyo tamaño depende de su valor de aptitud. 

En el muestreo universal estocástico (SUS), se hace un solo giro de la ruleta y se 
seleccionan varios individuos a la vez usando puntos de selección equidistantes

### Combinacion
### Mutacion
### Elitismo 