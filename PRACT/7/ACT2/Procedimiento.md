# Paso a paso de lo que hace el script min-char-rnn.py
```python

data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

```

Esto lee el contenido del input.txt y lo almacena como un string en data lo hace un conjunto osea que elimina todas las repeticiones, despues lo vuelve a hacer una lista y lo guarda en chrs que es el vocabulario completo

Despues mapea un cada caracter a su indice numerico y cada indice numerico a su caracter correspondiente
```python
# Para char_to_ix {'a': 0, 'b': 1, ...}
# Para ix_to_char {0: 'a', 1: 'b', ...}
```

## Hiperparametros
```python
# hiperparametros
hidden_size = 100 # tamaño de la capa de neuronas
seq_length = 25 # numero de pasos de desenrrollo
learning_rate = 1e-1 # tasa de aprendizaje
```

## Inicializacion de los pesos

```python
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # matrices de pesos para ajustar el modelo
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # matrices de pesos para ajustar el modelo
Why = np.random.randn(vocab_size, hidden_size)*0.01 # matrices de pesos para ajustar el modelo

# estos bias representan el sesgo para dos capas distintas de la red en el mismo paso de tiempo (t)
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias
```

## LossFun
```python
xs, hs, ys, ps = {}, {}, {}, {}
hs[-1] = np.copy(hprev) # Copia el estado oculto anterior
loss = 0

for t in range(len(inputs)):
  # Convierte el indice del caracter en un vector
  xs[t] = np.zeros((vocab_size,1))
  xs[t][inputs[t]] = 1

  # Calcula el nuevo estado oculto combinando la entrada y la memoria anterior aplicando tanh
  hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
  
  # Calcula las puntuaciones de salida
  ys[t] = np.dot(Why, hs[t]) + by 

  # Aplica softmax para obtener las probabilidades de los caracteres
  ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

  # Acumula la perdida de entropia cruzada
  loss += -np.log(ps[t][targets[t],0])

  # Calcula los gradientes que van hacia atras
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])

  for t in reversed(range(len(inputs))):
    
    # Calcula el gradiente de la capa Softmax, restando $1$ de la probabilidad del target.
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1

    # Acumula los gradientes para la matriz de salida y el sesgo de salida.
    dWhy += np.dot(dy, hs[t].T)
    dby += dy

    #propaga el gradiente dy a h_t, sumando el gradiente del futuro dhnext y multiplicando por la derivada de tanh
    dh = np.dot(Why.T, dy) + dhnext
    dhraw = (1 - hs[t] * hs[t]) * dh

    # Acumula los gradientes para el sesgo oculto y las matrices de transición.
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)

    #Calcula el gradiente que se enviará al estado oculto del paso de tiempo anterior h_t-1
    dhnext = np.dot(Whh.T, dhraw)

  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    #Recorta los gradientes para evitar el problema de los gradientes explosivos
    np.clip(dparam, -5, 5, out=dparam)
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
```


## Sample

### Se inicializa el vector x con el carácter semilla seed_ix que comienza la generación
```python
x = np.zeros((vocab_size, 1))
x[seed_ix] = 1
ixes = []
```

### Bucle
```python
for t in range(n):
  #Calcula el nuevo estado oculto h_t
  h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)

  # Calcula las puntuaciones de salida y las probabilidades Softmax
  y = np.dot(Why, h) + by
  p = np.exp(y) / np.sum(np.exp(y))

  # Selecciona el siguiente carácter aleatoriamente (ix) basándose en la distribución de probabilidades
  ix = np.random.choice(range(vocab_size), p=p.ravel())
  
  # El carácter recién seleccionado (ix) se convierte en la nueva entrada para el siguiente paso de tiempo, cerrando el ciclo de generación.
  x = np.zeros((vocab_size, 1))
  x[ix] = 1

  # Guarda el índice del carácter generado.
  ixes.append(ix)
return ixes

```


## Inicializacion 
```python
# n es la cantidad de epocas
# p es el puntero que marca la posicion de inicio de los datos
n, p = 0, 0

# Memoria de Adagrad pesos
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)

# Memoria de Adagrad bias
mbh, mby = np.zeros_like(bh), np.zeros_like(by)

# Perdida inicial para la iteracion 0
smooth_loss = -np.log(1.0/vocab_size)*seq_length
```

## Ciclo principal
### Comienzo del ciclo
```python
#Si el puntero mas la longitud de la secuencia se pasa del final de los datos o si es la primera iteración, se reinicia el entrenamiento.
if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
```

### Entradas y Objetivo

```python
#Se toma una secuencia de seq_length caracteres del texto, comenzando en p. Los caracteres se tokenizan (convierten a índices numéricos)
inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]


#Se toma una secuencia de seq_length caracteres del texto, comenzando en p+1. Los caracteres se tokenizan (convierten a índices numéricos)

targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
```
La diferencia entre el **<span style="color:red;">target</span>** y el **<span style="color:red;">input</span>** es que estamos moviendonos un caracter a la derecha en comparacion a la entrada.
El modelo va predecir el el siguiente caracter el **<span style="color:red;">t+1</span>** dado el caracter en la posicion **<span style="color:red;">t</span>**

### Cada 100 pasos se muestra una generacion de texto del modelo
```python
if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))
```

### Paso adelante y propagacion hacia atras y perdida
```python
# realiza la propagación hacia adelante (cálculo de la pérdida) y la propagación hacia atrás (cálculo de los gradientes)
loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)

#Se actualiza el promedio de la pérdida. El factor 0.999 da mucho peso a la pérdida histórica, asegurando que el reporte de pérdida sea estable.
smooth_loss = smooth_loss * 0.999 + loss * 0.001

# Se muestra la perdida que tiene el modelo
if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))
```
La funcion de lossFun devuelve la perdida, todos los gradientes y el ultimo estado oculto para ser usado como memoria de la siguiente secuencia

### Actualizacion de parametros con Adagrad
```python
for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                              [dWxh, dWhh, dWhy, dbh, dby], 
                              [mWxh, mWhh, mWhy, mbh, mby]):
  mem += dparam * dparam
  param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
```

El zip recorre simultáneamente los Parámetros **<span style="color:red;">param</span>**, sus Gradientes **<span style="color:red;">dparam</span>**, y la Memoria de Adagrad **<span style="color:red;">mem</span>**.

La memoria se actualiza sumando el cuadrado de los gradientes actuales.

Se actualiza el param de Adagrad

### Avanzar
```python
p += seq_length # mover el puntero del data
n += 1 # aumentar la iteracion
```
