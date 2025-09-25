# ENSEMBLE
Ensamblamos/ trabajamos con varios modelos.
### Voting ->
    Diferentes modelos, consultas a cada modelo y te quedas con la votacion por mayoria.

- Hard Voting (votación dura): Cada modelo individual emite su predicción como *clase discreta*.
    **La clase que obtiene la mayoría de los votos se elige como predicción final.**
- Soft voting -> Cada modelo predice *probabilidades* para cada clase. **Se suman o promedian las probabilidades y la clase con mayor probabilidad promedio es la elegida.**

#### Bagging con voting
- Bagging -> tengo un modelo que usa una bd que no es muy buena, agarra diferentes entrenamientos y con un tipo de modelo y se entrenan distintos modelos. (usa luego voting)

- Bagging RANDOM Forest (desicion Tree)
> Voting con Scikilearn

aleatoriedad en ram=ndom

-Booting -> agarra los datos que mejor resultado obtuvieron, analizando debilidades, para aprender de los errores. cuando el problema es de precision.