import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 200 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-2
n_epochs = 10000

# Parametros del modelo GRU
std = 0.005

# 1. Compuerta de Reinicio (r_gate)
# Nos dice cuanta informacion debe ignorarse osea reiniciarse
# Parámetros de la Compuerta de Reinicio (r)
Wxr = np.random.randn(hidden_size, vocab_size) * std
Whr = np.random.randn(hidden_size, hidden_size) * std
br = np.zeros((hidden_size, 1))

# 2. Compuerta de Actualización (z_gate)
# Decide cuanta informacion del pasado se va a conservar y cuanto del nuevo estado candidato se va a usar
# Parámetros de la Compuerta de Actualización (z)
Wxz = np.random.randn(hidden_size, vocab_size) * std
Whz = np.random.randn(hidden_size, hidden_size) * std
bz = np.zeros((hidden_size, 1))

# 3. Candidato a Estado Oculto (h_tilde)
# Es el nuevo estado propuesto que se combina con el estado anterior para formar el nuevo estado oculto
# Parámetros del Candidato a Estado Oculto (h_tilde)
Wxh = np.random.randn(hidden_size, vocab_size) * std
Whh = np.random.randn(hidden_size, hidden_size) * std
bh = np.zeros((hidden_size, 1))

# Parámetros de Salida (Output)
Why = np.random.randn(vocab_size, hidden_size) * std
by = np.zeros((vocab_size, 1))

# Funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def dtanh(y):
    return 1 - y * y

def lossFun(inputs, targets, hprev):
  """
  Función de pérdida (forward y backward pass) para GRU.
  """
  # 1. Inicializar diccionarios para guardar estados intermedios
  xs, hs, ys, ps = {}, {}, {}, {}
  r_gates, z_gates, h_tildes = {}, {}, {} # Estados internos de GRU
  
  hs[-1] = np.copy(hprev)
  loss = 0
  
  # --- FORWARD PASS (Cálculo de GRU) ---
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size, 1))
    xs[t][inputs[t]] = 1
    x = xs[t]
    h_prev = hs[t-1]

    # 1. Compuerta de Reinicio (r_gate)
    r_gate = sigmoid(np.dot(Wxr, x) + np.dot(Whr, h_prev) + br)
    r_gates[t] = r_gate

    # 2. Compuerta de Actualización (z_gate)
    z_gate = sigmoid(np.dot(Wxz, x) + np.dot(Whz, h_prev) + bz)
    z_gates[t] = z_gate

    # 3. Candidato a Estado Oculto (h_tilde)
    # r_gate * h_prev aplica el reinicio (elemento por elemento)
    h_tilde = np.tanh(np.dot(Wxh, x) + np.dot(Whh, (r_gate * h_prev)) + bh)
    h_tildes[t] = h_tilde

    # 4. Estado Oculto Final (h)
    # Combina el estado anterior (h_prev) y el candidato (h_tilde) usando z_gate
    hs[t] = (1 - z_gate) * h_prev + z_gate * h_tilde

    # 5. Output y Pérdida
    ys[t] = np.dot(Why, hs[t]) + by
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
    loss += -np.log(ps[t][targets[t], 0])

  # --- BACKWARD PASS ---
  
  # Inicializar gradientes de todos los nuevos parámetros
  dWxr, dWhr, dbr = np.zeros_like(Wxr), np.zeros_like(Whr), np.zeros_like(br)
  dWxz, dWhz, dbz = np.zeros_like(Wxz), np.zeros_like(Whz), np.zeros_like(bz)
  dWxh, dWhh, dbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(bh)
  dWhy, dby = np.zeros_like(Why), np.zeros_like(by)
  
  dhnext = np.zeros_like(hs[0])

  for t in reversed(range(len(inputs))):
    
    # 1. Gradiente de la Salida y dWhy, dby (igual que antes)
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1
    dWhy += np.dot(dy, hs[t].T)
    dby += dy

    # 2. Gradiente inicial para h[t]
    dh = np.dot(Why.T, dy) + dhnext # Suma de dhnext (de t+1) y el gradiente de la salida

    # 3. Retropropagación a través de la actualización
    h_prev = hs[t-1]
    z_gate = z_gates[t]
    h_tilde = h_tildes[t]

    # Gradiente en h_tilde y h_prev a través de la compuerta de actualización
    dh_tilde = dh * z_gate
    dh_prev_z = dh * (1 - z_gate)

    # Gradiente en la compuerta de actualización z_gate
    dz_gate = dh * (h_tilde - h_prev)
    dz_gate_raw = dsigmoid(z_gate) * dz_gate
    
    # 4. Retropropagación a través del candidato (h_tilde) y reinicio (r_gate)
    dh_tilde_raw = dtanh(h_tilde) * dh_tilde

    # Gradientes para Wxh, Whh, bh (candidato)
    dWxh += np.dot(dh_tilde_raw, xs[t].T)
    dWhh += np.dot(dh_tilde_raw, (r_gates[t] * h_prev).T)
    dbh += dh_tilde_raw

    # 5. Gradiente en la compuerta de reinicio (r_gate)
    dr_gate = dh_tilde_raw * np.dot(Whh, h_prev)
    dr_gate_raw = dsigmoid(r_gates[t]) * dr_gate
    
    # Gradientes para Wxr, Whr, br (reinicio)
    dWxr += np.dot(dr_gate_raw, xs[t].T)
    dWhr += np.dot(dr_gate_raw, h_prev.T)
    dbr += dr_gate_raw

    # 6. Gradientes para Wxz, Whz, bz (actualización)
    dWxz += np.dot(dz_gate_raw, xs[t].T)
    dWhz += np.dot(dz_gate_raw, h_prev.T)
    dbz += dz_gate_raw

    # 7. dhnext (para h[t-1])
    # Contribución 1: del estado oculto anterior
    #dh_prev_r = dh_tilde_raw * r_gates[t] * np.dot(Whh.T, np.ones_like(dr_gate_raw))
    dh_prev_r = np.dot(Whh.T, dh_tilde_raw) * r_gates[t]
    
    # Contribución 2: del gradiente de h a través de la compuerta de reinicio
    dh_prev_hr = np.dot(Whr.T, dr_gate_raw)
    
    # Contribución 3: del gradiente de h a través de la compuerta de actualización
    dh_prev_hz = np.dot(Whz.T, dz_gate_raw)

    # La contribución final a dhnext es la suma de todas las rutas hacia h[t-1]
    #dhnext = dh_prev_z + dh_prev_r + dh_prev_hr + dh_prev_hz
    dhnext = dh_prev_z + dh_prev_r + dh_prev_hr + dh_prev_hz
  
  # Recolectar todos los gradientes
  dparams = [dWxr, dWhr, dbr, dWxz, dWhz, dbz, dWxh, dWhh, dbh, dWhy, dby]
  
  # Clipping de gradientes
  for dparam in dparams:
    np.clip(dparam, -5, 5, out=dparam)

  # El retorno es la pérdida, todos los gradientes y el último estado oculto
  return loss, dWxr, dWhr, dbr, dWxz, dWhz, dbz, dWxh, dWhh, dbh, dWhy, dby, hs[len(inputs) - 1]

def sample(h_prev, seed_ix, n):
  """ 
  Muestra una secuencia de entradas del modelo GRU
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  h = np.copy(h_prev) # Iniciar con el estado previo (h_prev)

  for t in range(n):
    # Forward pass de GRU
    
    # Reinicio
    r_gate = sigmoid(np.dot(Wxr, x) + np.dot(Whr, h) + br)

    # Actualización
    z_gate = sigmoid(np.dot(Wxz, x) + np.dot(Whz, h) + bz)

    # Estado Oculto Candidato
    h_tilde = np.tanh(np.dot(Wxh, x) + np.dot(Whh, (r_gate * h)) + bh)

    # Estado Oculto Final
    h = (1 - z_gate) * h + z_gate * h_tilde
    
    # Salida y Muestreo
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
    
  return ixes

# --- BUCLE DE ENTRENAMIENTO ---

# Parámetros y variables de memoria de Adagrad
params = [Wxr, Whr, br, Wxz, Whz, bz, Wxh, Whh, bh, Why, by]
mparams = [np.zeros_like(p) for p in params] # Variables de memoria (m) para Adagrad

n, p = 0, 0
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss para la iteration 0

while True:
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1))
    p = 0
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # Muestra del modelo cada 100 iteraciones
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # reenvía caracteres seq_length a través de la red y recupera el gradiente
  loss_and_grads = lossFun(inputs, targets, hprev)
  loss = loss_and_grads[0]
  grads = loss_and_grads[1:-1]
  hprev = loss_and_grads[-1]

  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # Muestra el progreso
  
  # Actualización de parámetros con Adagrad
  for param, dparam, mem in zip(params, grads, mparams):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # actualización de Adagrad

  p += seq_length # moverse a la siguiente secuencia
  n += 1 # incrementar el contador de iteraciones
  n_epochs -= 1
  if n_epochs == -1:
      break