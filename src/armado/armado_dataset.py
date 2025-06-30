import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb


sellin = pl.read_csv("./../data/sell-in.txt", separator='\t')
print(sellin.head())

products_to_predict = pl.read_csv("./../data/product_id_apredecir201912.txt", separator='\t')
print(products_to_predict.head())

stocks = pl.read_csv("./../data/tb_stocks.txt", separator='\t')
print(stocks.head())

products = pl.read_csv("./../data/tb_productos.txt", separator='\t')
print(products.head())

masterdataset = sellin.join(products, on="product_id")
print(masterdataset.head())
print(masterdataset.shape)

print(masterdataset.group_by("periodo", "customer_id", "product_id").len() )


# # --- Solución para rellenar con ceros ---

print("Shape original del masterdataset:", masterdataset.shape)

# # Paso 1: Crear la "parrilla" completa de todas las combinaciones deseadas.
# # Primero, obtenemos todos los pares únicos de (customer_id, product_id) que existen en el dataset.
customer_product_pairs = masterdataset.select(["customer_id", "product_id"]).unique()

# # Luego, obtenemos todos los períodos únicos.
all_periods = masterdataset.select(pl.col("periodo").unique())

# # Creamos la parrilla completa con un cross join. Esto genera cada combinación posible
# # de (customer_id, product_id) con cada período.
grid = customer_product_pairs.join(all_periods, how="cross")


# # Paso 2: Unir la parrilla con los datos reales y rellenar nulos.
# # Hacemos un left join desde la parrilla hacia nuestros datos.
# # Las combinaciones que no existían en masterdataset tendrán nulos en las columnas de ventas.
data_filled = grid.join(masterdataset, on=["periodo", "customer_id", "product_id"], how="left")

# # Rellenamos los valores nulos de las ventas con 0.
# # También rellenamos los datos del producto (categoría, marca, etc.) que se volvieron nulos.
# # Usamos forward fill (ffill) agrupado por producto para propagar la información del producto.
data_filled = data_filled.with_columns(
    pl.col(["tn", "cust_request_tn", "cust_request_qty"]).fill_null(0),
    pl.col(['cat1', 'cat2', 'cat3', 'brand', 'sku_size', 'descripcion']).forward_fill().over("product_id")
)


# # Paso 3: Aplicar el filtro de lógica de negocio.
# # Queremos mantener una fila si (A) es una venta original, o (B) si en ese período,
# # el cliente estaba activo (compró algo) O el producto estaba activo (fue vendido a alguien).

# # Obtenemos la actividad de clientes por período.
customer_activity = masterdataset.select(["periodo", "customer_id"]).unique().with_columns(pl.lit(True).alias("customer_was_active"))

# # Obtenemos la actividad de productos por período.
product_activity = masterdataset.select(["periodo", "product_id"]).unique().with_columns(pl.lit(True).alias("product_was_active"))
product_activity.head(25)
# # Unimos estos "marcadores" de actividad a nuestro dataset rellenado.
final_dataset = data_filled \
    .join(customer_activity, on=["periodo", "customer_id"], how="left") \
    .join(product_activity, on=["periodo", "product_id"], how="left")

# # Filtramos según la lógica: mantener si la venta no es cero, O si el cliente estaba activo, O si el producto estaba activo.
final_dataset = final_dataset.filter(
    (pl.col("tn") > 0) |
    (pl.col("customer_was_active")) |
    (pl.col("product_was_active"))
)

# # Opcional: eliminar las columnas auxiliares
# final_dataset = final_dataset.drop(["customer_was_active", "product_was_active"])


print("Shape del dataset final rellenado:", final_dataset.shape)
print("Ejemplo de registros rellenados con cero:")
print(final_dataset.filter(pl.col("tn") == 0).head())


max_train_period=201910
"""Prepara datos de entrenamiento con series de tiempo"""
print(f"Preparando datos de entrenamiento hasta {max_train_period}")
# Agregar ventas por producto y periodo
ts_data = masterdataset.group_by('product_id', 'periodo').agg(pl.col("tn").sum())

# Ordenar por producto y periodo
ts_data = ts_data.sort(['product_id', 'periodo'])

# # Filtrar datos de entrenamiento
# train_data = ts_data['periodo'] <= max_train_period

# print(train_data)
# # Features a usar
# feature_cols = ['product_id', 'periodo', 'customer_id', 'plan_precios_cuidados']

# X = train_data('product_id', 'periodo', 'customer_id', 'plan_precios_cuidados')
# y = train_data['cust_request_tn']

# print(f"Registros de entrenamiento: {len(X)}")
# print(f"Features utilizadas: {len(feature_cols)}")
# print(f"Features: {feature_cols}")


# """Entrena modelos simples"""
# print("Entrenando modelos...")

# # Dividir datos manualmente para validación temporal
# # Usar últimos 6 meses como validación
# train_size = int(len(X) * 0.8)
# X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
# y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

# print(f"Entrenamiento: {len(X_train)} registros")
# print(f"Validación: {len(X_val)} registros")

# LightGBM
print("Entrenando LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=50000,
    learning_rate=0.01,
    max_depth=30,
    num_leaves=10,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
lgb_model


"""Evalúa modelos con la métrica personalizada"""
print("\n" + "="*50)
print("EVALUACIÓN DE MODELOS")
print("="*50)

# for name, model in lgb_model.items():
#     y_pred = model.predict(X_val)
#     y_pred = np.maximum(y_pred, 0)  # No predicciones negativas
    
#     mae = mean_absolute_error(y_val, y_pred)
#     custom_error = custom_forecast_error(y_val, y_pred)
    
#     print(f"\n{name.upper()} Results:")
#     print(f"MAE: {mae:.4f}")
#     print(f"Custom Forecast Error: {custom_error:.4f}")
#     print(f"Total Actual: {y_val.sum():.2f}")
#     print(f"Total Predicted: {y_pred.sum():.2f}")
