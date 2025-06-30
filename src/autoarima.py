import pandas as pd
import numpy as np
from pmdarima import auto_arima
from tqdm import tqdm

# Cargar datos
sellin = pd.read_csv("./../data/sell-in.txt", sep="\t")
products_to_predict = pd.read_csv("./../data/product_id_apredecir201912.txt")

# Agrupar ventas por producto y mes
ventas_mensuales = (
    sellin.groupby(["product_id", "periodo"])["tn"]
    .sum()
    .reset_index()
    .sort_values(["product_id", "periodo"])
)

# Convertir periodo a fecha para ordenar y filtrar
ventas_mensuales["periodo_date"] = pd.to_datetime(ventas_mensuales["periodo"].astype(str), format="%Y%m")

# Lista para guardar predicciones
predicciones = []

# Iterar por cada producto a predecir
for pid in tqdm(products_to_predict["product_id"]):
    serie = ventas_mensuales[ventas_mensuales["product_id"] == pid].copy()
    serie = serie.sort_values("periodo_date")

    # Limitar a los últimos 12 meses
    serie_ts = serie.set_index("periodo_date")["tn"].asfreq("MS").fillna(0)

    if len(serie_ts.dropna()) >= 6:
        try:
            modelo = auto_arima(
                serie_ts,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            pred = modelo.predict(n_periods=2)[-1]  # t+2
            pred = max(0, pred)
            if pred == 0:
                pred = serie_ts.mean()
        except:
            pred = serie_ts.mean()
    else:
        pred = serie_ts.mean()

    predicciones.append({"product_id": pid, "predicted_tn": round(pred, 5)})
    print(pred)

# Convertir a DataFrame
resultado = pd.DataFrame(predicciones)

# Guardar archivo
resultado.to_csv("predicciones_autoarima_feb2020.csv", index=False)
print("Predicciones guardadas en 'predicciones_autoarima_feb2020.csv'")
