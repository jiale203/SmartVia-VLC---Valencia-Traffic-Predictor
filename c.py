import pandas as pd

# 1. Cargar el dataset que ya teníamos combinado (tráfico + clima)
df_total = pd.read_csv('datos_combinados_limpios.csv')
df_total['fecha'] = pd.to_datetime(df_total['fecha'])

# 2. Cargar el nuevo dataset de festivos
df_festivos = pd.read_csv('DatasetFestivos.csv')

# 3. Convertir la columna 'Day' a datetime
# Usamos dayfirst=True porque el formato es Día/Mes/Año
df_festivos['Day'] = pd.to_datetime(df_festivos['Day'], dayfirst=True)

# 4. MERGE FINAL (Inner Join)
# Unimos por las columnas de fecha. 
# En df_total se llama 'fecha' y en df_festivos se llama 'Day'
df_final_completo = pd.merge(
    df_total, 
    df_festivos, 
    left_on='fecha', 
    right_on='Day', 
    how='inner'
)

# 5. Limpieza opcional: eliminar la columna 'Day' ya que es duplicada de 'fecha'
df_final_completo.drop(columns=['Day'], inplace=True)

# 6. Guardar el dataset maestro
df_final_completo.to_csv('dataset_final_maestro.csv', index=False)

print("¡Hecho! El dataset final solo contiene días donde hay información en los tres archivos.")
print(f"Total de filas finales: {len(df_final_completo)}")
print(df_final_completo.head())
