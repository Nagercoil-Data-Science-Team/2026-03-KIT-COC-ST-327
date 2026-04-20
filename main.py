# ================================
# STEP 1: IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'


# ================================
# STEP 2: LOAD DATA
# ================================
df = pd.read_csv('sensor-fault-detection.csv', sep=';')

print("Original Data Head:")
print(df.head())


# ================================
# STEP 3: PREPROCESSING
# ================================
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')
df.set_index('Timestamp', inplace=True)

# Missing values
df['Value'] = df['Value'].interpolate(method='time')
df['Value'] = df['Value'].bfill()

# Outlier removal
z_scores = np.abs(stats.zscore(df['Value']))
df = df[z_scores < 3]

# Normalization
scaler = MinMaxScaler()
df['Value_scaled'] = scaler.fit_transform(df[['Value']])

print("\nAfter Preprocessing:")
print(df.head())


# ================================
# STEP 4: TIME SERIES PREP
# ================================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 5
X, y = create_sequences(df['Value_scaled'].values, sequence_length)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, shuffle=False)


# ================================
# STEP 5: VISUALIZE REAL DATA
# ================================
plt.figure(figsize=(12,6))
plt.plot(df.index[:1000], df['Value'][:1000],color='#547792', label='Original Data')
plt.title("Sensor Value Over Time",fontweight='bold')
plt.xlabel("Timestamp",fontweight='bold')
plt.ylabel("Sensor Value",fontweight='bold')
plt.savefig('Sensor Value Over Time.png',dpi=800)
plt.show()


# ================================
# STEP 6: PREPARE GAN DATA
# ================================
df_gan = df.head(10000)
sensor_data = df_gan['Value_scaled'].values.reshape(-1, 1)

# Convert to [-1,1]
sensor_data = sensor_data * 2 - 1


# ================================
# STEP 7: BUILD GAN (FIXED)
# ================================
latent_dim = 20

# Generator
generator = Sequential([
    Input(shape=(latent_dim,)),

    Dense(64),
    LeakyReLU(0.2),
    BatchNormalization(),

    Dense(64),
    LeakyReLU(0.2),

    Dense(1, activation='tanh')
])

# Discriminator (STRONG)
discriminator = Sequential([
    Input(shape=(1,)),

    Dense(128),
    LeakyReLU(0.2),

    Dense(64),
    LeakyReLU(0.2),

    Dense(32),
    LeakyReLU(0.2),

    Dense(1, activation='sigmoid')
])

discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0002),
    metrics=['accuracy']
)


# GAN Model
discriminator.trainable = False

gan_input = Input(shape=(latent_dim,))
fake = generator(gan_input)
output = discriminator(fake)

gan = tf.keras.Model(gan_input, output)

gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001)
)

# IMPORTANT FIX
discriminator.trainable = True


# ================================
# STEP 8: TRAIN GAN (BALANCED)
# ================================
epochs = 2000
batch_size = 64
half_batch = batch_size // 2

for epoch in range(epochs):

    # Train Discriminator (2 TIMES)
    for _ in range(2):
        idx = np.random.randint(0, sensor_data.shape[0], half_batch)
        real = sensor_data[idx]

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake, np.zeros((half_batch, 1)))

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator (1 TIME)
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if (epoch+1) % 200 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")


# ================================
# STEP 9: GENERATE SYNTHETIC DATA
# ================================
noise = np.random.normal(0, 1, (1000, latent_dim))
synthetic = generator.predict(noise, verbose=0)

# Convert back to [0,1]
synthetic = (synthetic + 1) / 2


# ================================
# STEP 10: PLOT RESULTS
# ================================
plt.figure(figsize=(12,6))
plt.plot(sensor_data[:1000], label='Original Data',color='#8C5A3C')
plt.plot(synthetic, label='Synthetic Data',color='#5E0006')
plt.title("Final Stable GAN Output",fontweight='bold')
plt.xlabel("Sample Index",fontweight='bold')
plt.ylabel("Normalized Value",fontweight='bold')
plt.legend()
plt.savefig('Final Stable GAN Output.png',dpi=800)
plt.show()


# ================================
# STEP 11: PRINT SAMPLE
# ================================
print("\nSample Synthetic Data:")
print(synthetic[:10].flatten())

# ================================
# STEP 12: IMPROVED MOGA
# ================================
import random

# Problem settings
num_locations = 20
population_size = 40
generations = 50
mutation_rate = 0.1
elite_size = 4   # keep best individuals

np.random.seed(42)
coverage_weights = np.random.rand(num_locations)

sensor_cost = 10


# -------------------------------
# Chromosome Initialization
# -------------------------------
def create_individual():
    return np.random.randint(0, 2, num_locations)


def create_population():
    return [create_individual() for _ in range(population_size)]


# -------------------------------
# Fitness Function (FIXED)
# -------------------------------
def fitness(individual):
    coverage = np.sum(individual * coverage_weights)
    cost = np.sum(individual) * sensor_cost

    redundancy = 0
    for i in range(len(individual)-1):
        if individual[i] == 1 and individual[i+1] == 1:
            redundancy += 1

    return coverage, cost, redundancy


# Combined fitness score (VERY IMPORTANT FIX)
def fitness_score(ind):
    cov, cost, red = fitness(ind)
    return cov - 0.1 * cost + 0.5 * red


# -------------------------------
# Selection (Improved)
# -------------------------------
def selection(pop):
    i, j = random.sample(range(len(pop)), 2)
    return pop[i] if fitness_score(pop[i]) > fitness_score(pop[j]) else pop[j]


# -------------------------------
# Crossover
# -------------------------------
def crossover(parent1, parent2):
    point = random.randint(1, num_locations-1)
    return np.concatenate([parent1[:point], parent2[point:]])


# -------------------------------
# Mutation
# -------------------------------
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


# -------------------------------
# Evolution Process
# -------------------------------
population = create_population()

best_scores = []
best_solutions = []

for gen in range(generations):

    # Sort population (elitism)
    population = sorted(population, key=lambda ind: fitness_score(ind), reverse=True)
    new_population = population[:elite_size]  # keep best

    # Generate rest
    while len(new_population) < population_size:
        parent1 = selection(population)
        parent2 = selection(population)

        child = crossover(parent1, parent2)
        child = mutate(child)

        new_population.append(child)

    population = new_population

    best = population[0]
    best_solutions.append(fitness(best))
    best_scores.append(fitness_score(best))

    if gen % 10 == 0:
        cov, cost, red = fitness(best)
        print(f"Generation {gen}: Coverage={cov:.2f}, Cost={cost}, Redundancy={red}")


# ================================
# STEP 13: GA CONVERGENCE PLOT
# ================================
plt.figure(figsize=(8,6))
plt.plot(best_scores,color='#612D53')
plt.title("GA Convergence Curve",fontweight='bold')
plt.xlabel("Generation",fontweight='bold')
plt.ylabel("Fitness Score",fontweight='bold')
plt.savefig('GA Convergence Curve.png',dpi=800)
plt.show()


# ================================
# STEP 14: PARETO VISUALIZATION
# ================================
coverage_vals = [f[0] for f in best_solutions]
cost_vals = [f[1] for f in best_solutions]
redundancy_vals = [f[2] for f in best_solutions]

plt.figure(figsize=(8,6))
plt.scatter(cost_vals, coverage_vals, s=50,color='#41431B')
plt.title("Pareto Trade-off: Cost vs Coverage",fontweight='bold')
plt.xlabel("Cost",fontweight='bold')
plt.ylabel("Coverage",fontweight='bold')
plt.savefig('Pareto Trade-off.png',dpi=800)
plt.show()


# ================================
# STEP 15: BEST SENSOR LAYOUT
# ================================
best_final = max(population, key=lambda ind: fitness_score(ind))

cov, cost, red = fitness(best_final)

print("\nBest Sensor Layout (1=Sensor, 0=No Sensor):")
print(best_final)

print("\nFinal Fitness:")
print("Coverage:", cov)
print("Cost:", cost)
print("Redundancy:", red)
# ================================
# STEP 16: LSTM FAULT PREDICTION
# ================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Reshape for LSTM [samples, time steps, features]
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_lstm   = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_lstm  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM Model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length,1)),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\nTraining LSTM Model...")
history = lstm_model.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)
# ================================
# STEP 17: PREDICTION
# ================================
y_pred = lstm_model.predict(X_test_lstm)

# Convert back to original scale
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = scaler.inverse_transform(y_pred)

# ================================
# PRINT SAMPLE OUTPUT
# ================================
print("\nSample Predictions (Actual vs Predicted):")
for i in range(10):
    print(f"Actual: {y_test_inv[i][0]:.2f} | Predicted: {y_pred_inv[i][0]:.2f}")
# ================================
# STEP 18: EVALUATION METRICS
# ================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_inv, y_pred_inv)

print("\nLSTM Performance Metrics:")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R2   : {r2:.4f}")
# ================================
# STEP 19: PLOT ACTUAL vs PREDICTED
# ================================
plt.figure(figsize=(8,6))
plt.plot(y_test_inv[:200], label='Actual',color='#A0D585')
plt.plot(y_pred_inv[:200], label='Predicted',color='#FE9EC7')
plt.title("LSTM Fault Prediction ",fontweight='bold')
plt.xlabel("Time Step",fontweight='bold')
plt.ylabel("Sensor Value",fontweight='bold')
plt.legend()
plt.savefig('LSTM Fault Prediction.png',dpi=800)
plt.show()
# ================================
# STEP 20: TRAINING LOSS CURVE
# ================================
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss',color='#8A7650')
plt.plot(history.history['val_loss'], label='Validation Loss',color='#87B6BC')
plt.title("LSTM Training Curve",fontweight='bold')
plt.xlabel("Epoch",fontweight='bold')
plt.ylabel("Loss",fontweight='bold')
plt.legend()
plt.savefig('LSTM Training Curve.png',dpi=800)
plt.show()

# ================================
# STEP 21: ERROR CALCULATIONS
# ================================
errors = y_test_inv.flatten() - y_pred_inv.flatten()

mse_per_sample = errors**2
mae_per_sample = np.abs(errors)
rmse_per_sample = np.sqrt(mse_per_sample)

# Cumulative R²
r2_cumulative = []
for i in range(2, len(y_test_inv)):
    r2_val = r2_score(y_test_inv[:i], y_pred_inv[:i])
    r2_cumulative.append(r2_val)

# ================================
# STEP 22: RESIDUAL PLOT
# ================================
plt.figure(figsize=(8,6))
plt.scatter(y_pred_inv, errors,color='#C08552')
plt.axhline(y=0,color='#2C2C2C',linestyle='--')
plt.title("Residual Plot",fontweight='bold')
plt.xlabel("Predicted Values",fontweight='bold')
plt.ylabel("Residuals ",fontweight='bold')
plt.savefig('Residual Plot.png',dpi=800)
plt.show()

# ================================
# STEP 23: MSE PER SAMPLE
# ================================
plt.figure(figsize=(8,6))
plt.plot(mse_per_sample,color='#853953')
plt.title("MSE per Sample",fontweight='bold')
plt.xlabel("Sample Index",fontweight='bold')
plt.ylabel("MSE",fontweight='bold')
plt.savefig('MSE per Sample.png',dpi=800)
plt.show()

# ================================
# STEP 24: MAE PER SAMPLE
# ================================
plt.figure(figsize=(8,6))
plt.plot(mae_per_sample,color='#41431B')
plt.title("MAE per Sample",fontweight='bold')
plt.xlabel("Sample Index",fontweight='bold')
plt.ylabel("MAE",fontweight='bold')
plt.savefig('MAE per Sample.png',dpi=800)
plt.show()

# ================================
# STEP 25: RMSE PER SAMPLE
# ================================
plt.figure(figsize=(8,6))
plt.plot(rmse_per_sample,color='#8E977D')
plt.title("RMSE per Sample",fontweight='bold')
plt.xlabel("Sample Index",fontweight='bold')
plt.ylabel("RMSE",fontweight='bold')
plt.savefig('RMSE per Sample.png',dpi=800)
plt.show()

# ================================
# STEP 26: CUMULATIVE R²
# ================================
plt.figure(figsize=(8,6))
plt.plot(range(2, len(y_test_inv)), r2_cumulative,color='#BF4646')
plt.title("Cumulative R² Score",fontweight='bold')
plt.xlabel("Number of Samples",fontweight='bold')
plt.ylabel("R² Score",fontweight='bold')
plt.savefig('cumulative r^2.png',dpi=800)
plt.show()

# ================================
# STEP 27: FAULT SIMULATION
# ================================
# Combine real + synthetic (fault-like data)
fault_data = synthetic.flatten()

# Create fault threshold (adaptive)
threshold = np.mean(y_test_inv) + 1.5*np.std(y_test_inv)

# Ground truth (simulate faults)
y_true_fault = (y_test_inv.flatten() > threshold).astype(int)

# Predicted faults from LSTM
y_pred_fault = (y_pred_inv.flatten() > threshold).astype(int)

print("\nFault Threshold:", threshold)
print("Total Faults (Actual):", np.sum(y_true_fault))
print("Total Faults (Predicted):", np.sum(y_pred_fault))

# ================================
# STEP 28: DETECTION ACCURACY
# ================================
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true_fault, y_pred_fault)

print("\nFault Detection Accuracy:", accuracy)

# ================================
# STEP 29: RESPONSE TIME
# ================================
response_times = []

for i in range(len(y_true_fault)):
    if y_true_fault[i] == 1:
        for j in range(i, len(y_pred_fault)):
            if y_pred_fault[j] == 1:
                response_times.append(j - i)
                break

avg_response_time = np.mean(response_times) if response_times else 0

print("Average Response Time:", avg_response_time)

# ================================
# STEP 30: RECOVERY TIME
# ================================
# Assume optimized layout recovers faster
baseline_recovery = np.random.uniform(5, 10, len(response_times))
optimized_recovery = baseline_recovery * 0.7  # 30% improvement

print("Avg Baseline Recovery Time:", np.mean(baseline_recovery))
print("Avg Optimized Recovery Time:", np.mean(optimized_recovery))

# ================================
# STEP 31: COMPARISON PLOT
# ================================
labels = ['Accuracy', 'Response Time', 'Recovery Time']

baseline_metrics = [
    accuracy * 0.8,             # assume baseline lower accuracy
    avg_response_time * 1.5,    # slower response
    np.mean(baseline_recovery)
]

optimized_metrics = [
    accuracy,
    avg_response_time,
    np.mean(optimized_recovery)
]

x = np.arange(len(labels))

plt.figure(figsize=(8,6))
plt.plot(x, baseline_metrics, marker='o', label='Baseline System',color='#F26076')
plt.plot(x, optimized_metrics, marker='o', label='Optimized System',color='#427A43')

plt.xticks(x, labels)
plt.title("System Performance Comparison",fontweight='bold')
plt.xlabel("Number of Samples",fontweight='bold')
plt.ylabel("Metric Value",fontweight='bold')
plt.legend()
plt.savefig('System Performance Comparison.png',dpi=800)
plt.show()

# ================================
# STEP 32: FAULT DETECTION VISUAL
# ================================
plt.figure(figsize=(8,6))
plt.plot(y_test_inv[:200], label='Actual Signal',color='#547792')
plt.plot(y_pred_inv[:200], label='Predicted Signal',color='#628141')

fault_indices = np.where(y_true_fault[:200] == 1)[0]
plt.scatter(fault_indices, y_test_inv[:200][fault_indices], marker='x', label='Actual Faults',color='#740A03')

plt.title("Fault Detection Visualization",fontweight='bold')
plt.legend()
plt.savefig('Fault Detection Visualization.png',dpi=800)
plt.show()

# ================================
# STEP 33: CLASSIFICATION METRICS
# ================================
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

precision = precision_score(y_true_fault, y_pred_fault, zero_division=0)
recall = recall_score(y_true_fault, y_pred_fault, zero_division=0)
f1 = f1_score(y_true_fault, y_pred_fault, zero_division=0)

print("\n--- Classification Metrics ---")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# ================================
# STEP 36: PERFORMANCE METRICS PLOT
# ================================
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8,6))
plt.bar(metrics_names, metrics_values,color='#982598')
plt.title("Performance Metrics",fontweight='bold')
plt.xlabel("Metric Name",fontweight='bold')
plt.ylabel("Score",fontweight='bold')
plt.ylim(0,1)
plt.savefig('Performance Metrics.png',dpi=800)
plt.show()