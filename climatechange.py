import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.circuit import Parameter
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.circuit.library import RawFeatureVector

# Sample Disaster Data (Rainfall, Temp, Humidity -> Flood Risk)
data = np.array([[0.2, 0.5, 0.8], [0.9, 0.4, 0.3], [0.7, 0.8, 0.5], [0.1, 0.2, 0.9]])
labels = np.array([1, 0, 1, 0])  # 1 = High Disaster Risk, 0 = Low Risk

# Quantum Feature Encoding (Data ko Quantum Form me Convert Karna)
feature_map = RawFeatureVector(2**2)  # 4-Dimensional Quantum Encoding

# Variational Quantum Circuit (Quantum Neural Network)
qc = QuantumCircuit(2)
theta1 = Parameter('θ1')
theta2 = Parameter('θ2')
qc.ry(theta1, 0)
qc.ry(theta2, 1)
qc.cz(0, 1)  # Entanglement
qc.measure_all()

# Quantum ML Model - Variational Quantum Classifier
backend = Aer.get_backend('qasm_simulator')
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
vqc = VQC(quantum_kernel, optimizer=None)

# Train & Test Quantum Model
vqc.fit(data, labels)
prediction = vqc.predict(data)

print("Quantum Disaster Prediction:", prediction)

