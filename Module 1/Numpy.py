import numpy as np 

# zeros array
zeros_array = np.zeros(10)
print(zeros_array)

# ones arrary
ones_array = np.ones(10)
print(ones_array)

# constant array
constant_array = np.full(10,3)
print(constant_array)

# converting a list into array
a_list = [1,2,3,4,5]
li_to_ar = np.array(a_list)
print(li_to_ar)

# Generating Ranges of Numbers
range_array = np.arange(0,9)
print(range_array)

# Creating Arrays with Linear Spacing
lin_spc_ar = np.linspace(2,2,18)
print(lin_spc_ar)

# multidimensional arrays
zeros_matrix2 = np.zeros((5, 2))
print(zeros_matrix2)

ones_matrix2 = np.ones((2,3))
print(ones_matrix2)

constant_matrix = np.full((5,2),3)
print(constant_matrix)

# Indexing and Slicing Arrays
arr = np.array([[1,2,3],[4,5,6]])
print(arr)

# first row
first_row = arr[0]
first_col = arr[:,0]

# Generating Random Arrays
# set a seed to ensure reproducibility

np.random.seed(2)  # Set the seed
random_array = np.random.rand(5, 2)  # Generates random numbers between 0 and 1

# For random numbers from a normal distribution or integers within a range:
normal_distribution = np.random.randn(5, 2)
random_integers = np.random.randint(low=0, high=100, size=(5, 2))
print(random_integers)

# Pandas data frame
import pandas as pd
data = [
    ['Nissan', 'Stanza', 1991, 138, 4, 'MANUAL', 'sedan', 2000],
    ['Hyundai', 'Sonata', 2017, None, 4, 'AUTOMATIC', 'Sedan', 27150],
    ['Lotus', 'Elise', 2010, 218, 4, 'MANUAL', 'convertible', 54990],
    ['GMC', 'Acadia',  2017, 194, 4, 'AUTOMATIC', '4dr SUV', 34450],
    ['Nissan', 'Frontier', 2017, 261, 6, 'MANUAL', 'Pickup', 32340],
]
  
columns = [
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Vehicle_Style', 'MSRP'
] 

df = pd.DataFrame(data, columns = columns)
df

