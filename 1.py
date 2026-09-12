import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
housing = pd.read_csv('C:\\Documentos\\2SEM2026\\ArtificialIntelligence\\Datasets\\california_housing_prices.csv'
)

# Verificar que la columna existe
print(housing.columns)

# Obtener mínimo y máximo
minimo = housing["housing_median_age"].min()
maximo = housing["housing_median_age"].max()

# Calcular el rango
rango = maximo - minimo
media_population = housing["population"].mean()
porcentaje_faltantes = housing["total_bedrooms"].isnull().mean() * 100
correlacion = housing["total_rooms"].corr(housing["total_bedrooms"])

# Calcular correlaciones de las variables numéricas
correlaciones = housing.corr(numeric_only=True)["median_house_value"]

# Eliminar la correlación de median_house_value consigo misma
correlaciones = correlaciones.drop("median_house_value")

# Mostrar todas las correlaciones
print("Correlaciones con median_house_value:")
print(correlaciones)

# Encontrar la correlación más fuerte en valor absoluto
variable_mayor = correlaciones.abs().idxmax()

print("\nVariable con la correlación más fuerte:", variable_mayor)
print("Valor de correlación:", correlaciones[variable_mayor])

print("Correlación entre total_rooms y total_bedrooms:", correlacion)

print("Porcentaje de valores faltantes:", porcentaje_faltantes, "%")
print("Valor medio de population:", media_population)

# Mostrar resultados
print("Valor mínimo:", minimo)
print("Valor máximo:", maximo)
print("Intervalo de valores:", minimo, "a", maximo)
print("Rango:", rango)