import pandas as pd

# Parámetro configurable: últimos N meses a considerar (ej. 3, 6, 9, 12)
n_meses = 24  # Cambiar este valor para ajustar

# Cargar datos
sellin = pd.read_csv("./../data/sell-in.txt", sep="\t")
products_to_predict = pd.read_csv("./../data/product_id_apredecir201912.txt")

# Agrupar ventas por producto y mes (suma total del mes)
ventas_mensuales = (
    sellin.groupby(["product_id", "periodo"])["tn"]
    .sum()
    .reset_index(name="tn_mensual")
)

# Ordenar por periodo
ventas_mensuales = ventas_mensuales.sort_values(["product_id", "periodo"])

# Filtrar los últimos N meses por producto
ventas_mensuales["rank"] = ventas_mensuales.groupby("product_id")["periodo"].rank(method="first", ascending=False)
ventas_mensuales_filtradas = ventas_mensuales.groupby("product_id").tail(n_meses)

# Calcular el promedio solo sobre los últimos N meses
promedios_recientes = (
    ventas_mensuales_filtradas.groupby("product_id")["tn_mensual"]
    .mean()
    .reset_index()
    .rename(columns={"tn_mensual": "predicted_tn"})
)

# Filtrar productos a predecir
resultado = products_to_predict.merge(promedios_recientes, on="product_id", how="left")

# Reemplazar NaNs por 0 (sin historial suficiente)
resultado["predicted_tn"] = resultado["predicted_tn"].fillna(0)

# Guardar resultado
resultado.to_csv(f"predicciones_baseline_{n_meses}meses.csv", index=False)
print(f"Predicciones guardadas en 'predicciones_baseline_{n_meses}meses.csv'")
