import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
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
ventas_mensuales["ds"] = pd.to_datetime(ventas_mensuales["periodo"].astype(str), format="%Y%m")
ventas_mensuales.rename(columns={"product_id": "unique_id", "tn": "y"}, inplace=True)

# Lista para guardar predicciones
predicciones = []

# Iterar por cada producto a predecir
for pid in tqdm(products_to_predict["product_id"]):
    serie = ventas_mensuales[ventas_mensuales["unique_id"] == pid].copy()
    serie = serie.sort_values("ds")
    serie_last_12 = serie.tail(12)

    if len(serie_last_12) >= 6:
        try:
            sf = StatsForecast(df=serie_last_12, models=[AutoARIMA(season_length=12)], freq='MS')
            forecast_df = sf.forecast(h=2)
            pred = forecast_df.iloc[-1]["AutoARIMA"]
            pred = max(0, pred)
            if pred == 0:
                pred = serie_last_12["y"].mean()
        except:
            pred = serie_last_12["y"].mean()
    else:
        pred = serie_last_12["y"].mean()

    predicciones.append({"product_id": pid, "predicted_tn": round(pred, 5)})

# Convertir a DataFrame
resultado = pd.DataFrame(predicciones)

# Extraer ventas reales de diciembre 2019
ventas_dic = ventas_mensuales[ventas_mensuales["ds"] == "2019-12-01"]
ventas_dic = ventas_dic[["unique_id", "y"]].rename(columns={"unique_id": "product_id", "y": "real_tn"})

# Unir predicciones con ventas reales
comparacion = resultado.merge(ventas_dic, on="product_id", how="left")

# Graficar predicciones vs realidad
plt.figure(figsize=(10, 6))
plt.scatter(comparacion["predicted_tn"], comparacion["real_tn"], alpha=0.6)
plt.xlabel("Predicción Febrero 2020")
plt.ylabel("Realidad Diciembre 2019")
plt.title("Predicción vs Realidad por producto")
plt.grid(True)
plt.axline((0, 0), slope=1, color='red', linestyle='--')
plt.tight_layout()
plt.savefig("comparacion_pred_vs_real.png")
plt.show()

# Guardar archivo
resultado.to_csv("predicciones_autoarima_feb2020.csv", index=False)
print("Predicciones guardadas en 'predicciones_autoarima_feb2020.csv'")
