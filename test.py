import sympy as sp
import numpy as np

# Define symbolic variables in SymPy
a, t, hover_t, t_len = sp.symbols('a t hover_t t_len')

# Define the sympy expression with Heaviside
XRef_expr = sp.Heaviside(t - hover_t) * 3 * sp.sin(4 * sp.pi * a * t / t_len)

# Custom Heaviside function for numpy that supports arrays
def heaviside(x, zero_value=0):
    return np.where(x > 0, 1, 0)

# Lambdify with custom Heaviside function and numpy as backend
# Vectorize the function to handle array inputs element-wise
XRef = sp.lambdify((a, t, hover_t, t_len), XRef_expr, {'Heaviside': heaviside, 'numpy': np})
XRef_vectorized = np.vectorize(XRef)  # Vectorize to handle element-wise operations

# Example arrays as inputs
self_a = 1   # Array for 'a'
self_t = np.linspace(0, 10, 100)      # Array for 't'
self_hover_t = 2                      # Scalar for hover_t
self_t_len = 10                       # Scalar for t_len

# Broadcasting to evaluate XRef with array inputs using np.vectorize
XRef_output = XRef_vectorized(self_a, self_t, self_hover_t, self_t_len)
print(XRef_output)
print("Shape of XRef output:", XRef_output.shape)