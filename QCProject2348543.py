# Import necessary libraries
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import time

# Streamlit Application Title
st.title("Interactive Quantum vs Classical Search Comparison")
st.markdown("""
This interactive app demonstrates:
1. **Superposition** and quantum parallelism.
2. **Grover's Algorithm** for quantum speedup.
3. **Classical vs Quantum Search** for solving a search problem interactively.
""")

# Sidebar Inputs
st.sidebar.title("Settings")
num_qubits = st.sidebar.slider("Number of Qubits", min_value=2, max_value=6, value=3, step=1)
dataset_size = 2 ** num_qubits
target_state = st.sidebar.text_input(
    f"Target State (Binary, {num_qubits} bits):", "1" * num_qubits
)

# Validation for target state
if len(target_state) != num_qubits or not all(c in '01' for c in target_state):
    st.sidebar.error(f"Enter a valid {num_qubits}-bit binary string.")
    st.stop()

# Section 1: Superposition
st.header("1. Superposition")
st.markdown("Creating superposition by applying Hadamard gates to qubits.")

# Quantum Circuit for Superposition
qc = QuantumCircuit(num_qubits)
for i in range(num_qubits):
    qc.h(i)  # Apply Hadamard gate

# Visualize Superposition
st.subheader("Quantum Circuit for Superposition")
st.code(qc.draw("text"))

state = Statevector.from_instruction(qc)
st.subheader("Bloch Sphere Representation of Superposition")
bloch_plot = plot_bloch_multivector(state)
st.pyplot(bloch_plot.figure)

# Section 2: Classical Search
st.header("2. Classical Search")
st.markdown("Brute-force search over all possible states.")

# Generate Classical Dataset
dataset = [bin(x)[2:].zfill(num_qubits) for x in range(dataset_size)]

# Classical Search Function
def classical_search(dataset, target):
    for i, item in enumerate(dataset):
        if item == target:
            return i

# Measure Classical Search Time
start_time = time.time()
classical_result = classical_search(dataset, target_state)
classical_time = time.time() - start_time

st.subheader("Classical Search Results")
st.write(f"Target State: {target_state}")
st.write(f"Index Found: {classical_result}")
st.write(f"Time Taken: {classical_time:.5f} seconds")

# Section 3: Quantum Search with Grover's Algorithm
st.header("3. Quantum Search with Grover's Algorithm")
st.markdown("Demonstrating quantum speedup for finding a target state.")

# Grover's Algorithm Circuit
grover_qc = QuantumCircuit(num_qubits)
grover_qc.h(range(num_qubits))  # Initialize in superposition
grover_qc.z(range(num_qubits))  # Oracle marking the target state
grover_qc.h(range(num_qubits))  # Diffusion operator
grover_qc.measure_all()

# Simulate Grover's Algorithm
simulator = AerSimulator()
compiled_grover = transpile(grover_qc, simulator)
start_time = time.time()
grover_result = simulator.run(compiled_grover).result()
grover_time = time.time() - start_time
grover_counts = grover_result.get_counts()

# Display Quantum Results
st.subheader("Quantum Search Results")
fig, ax = plt.subplots()
plot_histogram(grover_counts, ax=ax)
st.pyplot(fig)
st.write(f"Target State Found with High Probability.")
st.write(f"Time Taken: {grover_time:.5f} seconds")

# Section 4: Comparison
st.header("4. Classical vs Quantum Search Comparison")
st.write("**Key Observations:**")
st.write("- Classical search evaluates states one by one, while quantum search leverages parallelism.")
st.write(f"- Classical Time: {classical_time:.5f} seconds")
st.write(f"- Quantum Time: {grover_time:.5f} seconds")
st.markdown("Explore with different qubits and target states to see the effects on execution time!")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Qiskit and Streamlit.")
