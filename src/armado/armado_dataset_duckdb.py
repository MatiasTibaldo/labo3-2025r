import duckdb

# Establish a DuckDB connection
con = duckdb.connect(database=':memory:', read_only=False)

sellin = con.execute("""
    SELECT * FROM read_csv_auto('./../../data/sell-in.txt', delim='\t', header=True)
""").fetchdf()

products_to_predict = con.execute("""
    SELECT * FROM read_csv_auto('./../../data/product_id_apredecir201912.txt', delim='\t', header=True)
""").fetchdf()

stocks = con.execute("""
    SELECT * FROM read_csv_auto('./../../data/tb_stocks.txt', delim='\t', header=True)
""").fetchdf()

products = con.execute("""
    SELECT * FROM read_csv_auto('./../../data/tb_productos.txt', delim='\t', header=True)
""").fetchdf()


print(products_to_predict.head())
print(stocks.head())
print(products.head())
print(sellin.head())


# Registramos los DataFrames como tablas temporales
con.register('sellin', sellin)
con.register('products', products)

# Realizamos el join
joined_df = con.execute("""
    SELECT s.*, p.*
    FROM sellin s
    JOIN products p
    ON s.product_id = p.product_id
""").fetchdf()
joined_df = joined_df.drop('product_id_1', axis=1)
print(joined_df.head())

# Realizamos el join
con.execute("CREATE OR REPLACE TABLE joined_table AS SELECT * FROM joined_df")

# 1. Determinar el período de existencia de cada producto
con.execute("""
CREATE OR REPLACE TABLE producto_periodo AS
SELECT 
    product_id,
    MIN(periodo) AS periodo_inicio,
    MAX(periodo) AS periodo_fin
FROM joined_table
GROUP BY product_id;
""")

# 2. Determinar el período de existencia de cada cliente
con.execute(
    """
    CREATE OR REPLACE TABLE cliente_periodo AS
SELECT 
    customer_id,
    MIN(periodo) AS periodo_inicio,
    MAX(periodo) AS periodo_fin
FROM joined_table
GROUP BY customer_id;

    """
)
# 3. Buscar todos los periodos
con.execute(
    """
CREATE OR REPLACE TABLE todos_los_periodos AS
SELECT DISTINCT(periodo) AS periodo
FROM joined_table
ORDER BY periodo ASC ;

    """
)

# 4. Detectar combinaciones que NO existen en joined_table
## TODO puse 201912 para ver los productos muertos y se puede hacer lo mismo con clientes
con.execute(
    """
CREATE OR REPLACE TABLE posibles_combinaciones AS
SELECT 
    p.product_id,
    c.customer_id,
    t.periodo
FROM producto_periodo p
JOIN cliente_periodo c ON TRUE
JOIN todos_los_periodos t ON t.periodo BETWEEN p.periodo_inicio AND 201912
                          AND t.periodo BETWEEN c.periodo_inicio AND c.periodo_fin; 
    """
)

# 5. Detectar combinaciones que NO existen en joined_table
con.execute(
    """
CREATE OR REPLACE TABLE faltantes AS
SELECT 
    pc.customer_id,
    pc.product_id,
    pc.periodo,
    0.0 AS tn,
FROM posibles_combinaciones pc
LEFT JOIN joined_table jt
  ON jt.customer_id = pc.customer_id 
 AND jt.product_id = pc.product_id 
 AND jt.periodo = pc.periodo
WHERE jt.customer_id IS NULL;
    """
)

# 6. combinar
con.execute("""
CREATE OR REPLACE TABLE faltantes_completos AS
SELECT 
    f.customer_id,
    f.product_id,
    f.periodo,
    f.tn,
    p.cat1,
    p.cat2,
    p.cat3,
    p.brand,
    p.sku_size,
    0 plan_precios_cuidados,
    0 cust_request_qty,
    0 cust_request_tn,
    p.descripcion
FROM faltantes f
LEFT JOIN products p ON f.product_id = p.product_id;
""")

con.execute(
    """CREATE OR REPLACE TABLE dataset_final AS
SELECT 
    customer_id,
    product_id,
    periodo,
    tn,
    cat1,
    cat2,
    cat3,
    brand,
    sku_size,
    plan_precios_cuidados,
    cust_request_qty,
    cust_request_tn,
    descripcion
FROM joined_table

UNION ALL

SELECT 
    customer_id,
    product_id,
    periodo,
    tn,
    cat1,
    cat2,
    cat3,
    brand,
    sku_size,
    plan_precios_cuidados,
    cust_request_qty,
    cust_request_tn,
    descripcion
FROM faltantes_completos;
"""
)

result = con.execute(
    """select * from dataset_final"""
).fetchdf()
result.head()
result.shape

con.execute(
    """
CREATE OR REPLACE TABLE dataset_final_features AS
SELECT 
    *,
    CAST(periodo / 100 AS INTEGER) AS year,
    CAST(periodo % 100 AS INTEGER) AS month,
    CASE 
        WHEN (periodo % 100) IN (1,2,3) THEN 1
        WHEN (periodo % 100) IN (4,5,6) THEN 2
        WHEN (periodo % 100) IN (7,8,9) THEN 3
        WHEN (periodo % 100) IN (10,11,12) THEN 4
        ELSE NULL
    END AS quarter
FROM dataset_final;
    """
)

