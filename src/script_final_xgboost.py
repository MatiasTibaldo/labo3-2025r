import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from xgboost.callback import EarlyStopping
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Optuna no disponible. Se usarán parámetros predeterminados.")
    OPTUNA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet no disponible. Se omitirán features de Prophet.")
    PROPHET_AVAILABLE = False

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import dtw
    DTW_AVAILABLE = True
except ImportError:
    print("tslearn no disponible. Se omitirá clustering DTW.")
    DTW_AVAILABLE = False

import pickle
from datetime import datetime, timedelta
from scipy import stats
import gc

class SalesPredictor:
    def __init__(self):
        # self.scalers = {}
        self.label_encoders = {}
        self.prophet_models = {}
        self.xgb_model = None
        self.feature_names = []
        self.cluster_model = None
        
    def load_data(self, sell_in_path, products_path, stocks_path, predict_products_path):
        """Carga todos los datasets necesarios"""
        self.sell_in = pd.read_csv(sell_in_path, sep='\t', encoding='utf-8')
        self.products = pd.read_csv(products_path, sep='\t')
        self.stocks = pd.read_csv(stocks_path, sep='\t')
        self.predict_products = pd.read_csv(predict_products_path, sep='\t')
        
        # Convertir periodo a datetime
        self.sell_in['periodo'] = pd.to_datetime(self.sell_in['periodo'].astype(str), format='%Y%m')
        self.stocks['periodo'] = pd.to_datetime(self.stocks['periodo'].astype(str), format='%Y%m')
        
        print(f"Datos cargados: {len(self.sell_in)} registros de ventas")
        print(f"Productos únicos: {self.sell_in['product_id'].nunique()}")
        print(f"Productos a predecir: {len(self.predict_products)}")
        
    def create_time_features(self, df):
        """Crear features temporales"""
        df = df.copy()
        df['año'] = df['periodo'].dt.year
        df['mes'] = df['periodo'].dt.month
        df['quarter'] = df['periodo'].dt.quarter
        df['dias_mes'] = df['periodo'].dt.days_in_month
        
        # Continuidad temporal con seno y coseno
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        return df
    
    def create_lag_features(self, df, target_col='tn', max_lag=30):
        """Crear features de lag y delta lag"""
        df = df.copy()
        
        # Ordenar por product_id, customer_id y periodo
        df = df.sort_values(['product_id', 'customer_id', 'periodo'])
        
        # Crear lags
        for lag in range(1, max_lag + 1):
            df[f'lag_{lag}'] = df.groupby(['product_id', 'customer_id'])[target_col].shift(lag)
        
        # Crear delta lags (diferencias entre lags)
        for lag in range(1, max_lag):
            df[f'delta_lag_{lag}'] = df[f'lag_{lag}'] - df[f'lag_{lag+1}']
            
        return df
            
    def create_rolling_features(self, df, target_col='tn', max_window=36):
        """Crear features de medias móviles, mínimos y máximos"""
        df = df.copy()
        df = df.sort_values(['product_id', 'customer_id', 'periodo']).reset_index(drop=True)
        
        for window in range(2, max_window + 1):
            # Medias móviles
            print(f"Calculando rolling mean para ventana {window} meses...")
            rolling_mean = df.groupby(['product_id', 'customer_id'])[target_col].rolling(
                window=window, min_periods=1).mean().values
            df[f'rolling_mean_{window}'] = rolling_mean
            
            # Calcular rolling min y max
            print(f"Calculando rolling min y max para ventana {window} meses...")
            rolling_min = df.groupby(['product_id', 'customer_id'])[target_col].rolling(
                window=window, min_periods=1).min().values
            rolling_max = df.groupby(['product_id', 'customer_id'])[target_col].rolling(
                window=window, min_periods=1).max().values
            
            # Mínimos booleanos
            df[f'is_min_{window}'] = (rolling_min == df[target_col].values).astype(int)
            
            # Máximos booleanos  
            df[f'is_max_{window}'] = (rolling_max == df[target_col].values).astype(int)
        
        return df
    
    def create_prophet_features(self, df, target_col='tn'):
        """Crear features usando Prophet"""
        if not PROPHET_AVAILABLE:
            print("Prophet no disponible, omitiendo features de Prophet")
            return df
            
        prophet_features = []
        
        for product_id in df['product_id'].unique():
            product_data = df[df['product_id'] == product_id].copy()
            
            # Agregar por periodo para Prophet
            product_agg = product_data.groupby('periodo')[target_col].sum().reset_index()
            product_agg.columns = ['ds', 'y']
            
            if len(product_agg) < 10:  # Mínimo de datos para Prophet
                continue
                
            try:
                # Configurar y entrenar Prophet
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.1,
                    seasonality_prior_scale=0.1
                )
                
                model.fit(product_agg)
                
                # Generar predicciones
                forecast = model.predict(product_agg[['ds']])
                
                # Crear features
                prophet_df = product_agg[['ds']].copy()
                prophet_df['product_id'] = product_id
                prophet_df['prophet_trend'] = forecast['trend']
                prophet_df['prophet_seasonal'] = forecast['yearly']
                prophet_df['prophet_yhat'] = forecast['yhat']
                prophet_df['prophet_yhat_lower'] = forecast['yhat_lower']
                prophet_df['prophet_yhat_upper'] = forecast['yhat_upper']
                
                prophet_df.rename(columns={'ds': 'periodo'}, inplace=True)
                prophet_features.append(prophet_df)
                
            except Exception as e:
                print(f"Error en Prophet para producto {product_id}: {e}")
                continue
        
        if prophet_features:
            prophet_df = pd.concat(prophet_features, ignore_index=True)
            df = df.merge(prophet_df, on=['periodo', 'product_id'], how='left')
        
        return df
    
    def create_dtw_clusters(self, df, target_col='tn', n_clusters=50):
        """Crear clusters usando Dynamic Time Warping"""
        if not DTW_AVAILABLE:
            print("tslearn no disponible, omitiendo clustering DTW")
            df['dtw_cluster_50'] = -1
            return df
            
        try:
            # Preparar series temporales por producto
            product_series = []
            product_ids = []
            
            for product_id in df['product_id'].unique():
                product_data = df[df['product_id'] == product_id].copy()
                
                # Agregar por periodo
                series = product_data.groupby('periodo')[target_col].sum().values
                
                # Filtrar series con suficientes datos
                if len(series) >= 12:  # Al menos 12 meses
                    product_series.append(series)
                    product_ids.append(product_id)
            
            if len(product_series) < 10:
                print("No hay suficientes series para clustering DTW")
                df['dtw_cluster_50'] = -1
                return df
            
            # Normalizar series para DTW
            normalized_series = []
            for series in product_series:
                if np.std(series) > 0:
                    normalized = (series - np.mean(series)) / np.std(series)
                else:
                    normalized = series
                normalized_series.append(normalized)
            
            # Aplicar clustering DTW
            self.cluster_model = TimeSeriesKMeans(
                n_clusters=min(n_clusters, len(normalized_series)),
                metric="dtw",
                max_iter=10,
                random_state=42,
                n_jobs=-1
            )
            
            clusters = self.cluster_model.fit_predict(normalized_series)
            
            # Crear DataFrame con clusters
            cluster_df = pd.DataFrame({
                'product_id': product_ids,
                'dtw_cluster_50': clusters
            })
            self.product_cluster_map = cluster_df.set_index('product_id')['dtw_cluster_50']

            # Merge con datos principales
            df = df.merge(cluster_df, on='product_id', how='left')
            df['dtw_cluster_50'] = df['dtw_cluster_50'].fillna(-1)
            
            print(f"DTW clustering completado: {len(np.unique(clusters))} clusters")
            
        except Exception as e:
            print(f"Error en DTW clustering: {e}")
            df['dtw_cluster_50'] = -1
            
        return df
    
    def add_external_features(self, df):
        """Agregar variables exógenas simuladas"""
        # Simular variables exógenas (en la práctica, cargarías datos reales)
        np.random.seed(42)
        
        periodos_unicos = df['periodo'].unique()
        external_data = []
        
        for periodo in periodos_unicos:
            external_data.append({
                'periodo': periodo,
                'dolar_cotizacion': np.random.normal(50, 10),  # Simular cotización dólar
                'ipc': np.random.normal(100, 5),  # Simular IPC
                'receso_escolar': 1 if pd.to_datetime(periodo).month in [7, 12, 1] else 0,
                'mes_feriados': 1 if pd.to_datetime(periodo).month in [5, 7, 12] else 0,
                'anomalia_politica': np.random.choice([0, 1], p=[0.9, 0.1])
            })
        
        external_df = pd.DataFrame(external_data)
        df = df.merge(external_df, on='periodo', how='left')
        
        return df
    
    def create_ratio_target(self, df, target_col='tn'):
        """Crear target como ratio (tn / tn+2)"""
        df = df.copy()
        df = df.sort_values(['product_id', 'customer_id', 'periodo'])
        
        # Calcular suma de próximos 2 periodos
        df['tn_next_2'] = (
            df.groupby(['product_id', 'customer_id'])[target_col].shift(-1).fillna(0) +
            df.groupby(['product_id', 'customer_id'])[target_col].shift(-2).fillna(0)
        )
        
        # Crear ratio, evitando división por cero
        df['target_ratio'] = df[target_col] / (df['tn_next_2'] + 1e-8)
        df['target_ratio'] = df['target_ratio'].replace([np.inf, -np.inf], 0)
        
        return df
    
    def handle_missing_values(self, df):
        """Tratamiento de valores faltantes"""
        # Para variables numéricas, mantener NaN (como indica el documento)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Para variables categóricas, usar 'unknown'
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if col not in ['periodo']:
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def normalize_data(self, df, target_col='tn', fit_scalers=True):
        """Normalización estándar por producto"""
        df = df.copy()
        
        # Columnas relacionadas con toneladas para normalizar
        tonnage_cols = [col for col in df.columns if 
                       col.startswith(('tn', 'lag_', 'delta_lag_', 'rolling_mean_', 'prophet_'))]
        
        if fit_scalers:
            self.scalers = {}
        
        for product_id in df['product_id'].unique():
            product_mask = df['product_id'] == product_id
            
            if fit_scalers:
                scaler = StandardScaler()
                product_data = df.loc[product_mask, tonnage_cols]
                
                # Solo ajustar si hay datos suficientes
                if len(product_data) > 1 and not product_data.isna().all().all():
                    scaler.fit(product_data.fillna(0))
                    self.scalers[product_id] = scaler
                    
                    # Aplicar transformación
                    df.loc[product_mask, tonnage_cols] = scaler.transform(
                        product_data.fillna(0))
            else:
                # Aplicar transformación existente
                if product_id in self.scalers:
                    product_data = df.loc[product_mask, tonnage_cols]
                    df.loc[product_mask, tonnage_cols] = self.scalers[product_id].transform(
                        product_data.fillna(0))
        
        return df
    
    def encode_categorical_features(self, df, fit_encoders=True):
        """One-hot encoding para variables categóricas"""
        df = df.copy()
        
        categorical_cols = ['cat1', 'cat2', 'cat3', 'brand']
        
        if fit_encoders:
            self.label_encoders = {}
        
        for col in categorical_cols:
            if col in df.columns:
                if fit_encoders:
                    le = LabelEncoder()
                    df[col] = df[col].fillna('unknown')
                    le.fit(df[col])
                    self.label_encoders[col] = le
                
                # Aplicar encoding
                if col in self.label_encoders:
                    df[col] = df[col].fillna('unknown')
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
        
    def calculate_sample_weights(self, df, target_col='tn'):
        """Calcular pesos de muestra para penalizar productos con más toneladas (versión robusta)"""
        print("Calculando pesos de muestra...")
        
        # 1. Calcular la suma de toneladas por producto
        weights = df.groupby('product_id')[target_col].sum()
        
        # 2. Calcular peso inverso. Esto se mantiene siempre positivo si la suma es positiva.
        weights = 1 / (weights + 1e-8)
        
        # 3. Mapear los pesos de vuelta al DataFrame original
        sample_weights = df['product_id'].map(weights)
                
        # Rellenar cualquier NaN que resulte del mapeo con un peso neutral (1.0)
        sample_weights = sample_weights.fillna(1.0)
        
        # Asegurar que ningún peso sea cero o negativo. Usamos clip para establecer
        # un "suelo" muy pequeño pero positivo.
        sample_weights = sample_weights.clip(lower=1e-8)
        
        # final_weights = (sample_weights / sample_weights.sum()) * len(sample_weights)

        print("Pesos de muestra calculados y validados.")
        return sample_weights.values
    
    def prepare_features(self, fit_mode=True):
        """Preparar todas las features siguiendo el pipeline completo"""
        print("Iniciando feature engineering...")
        
        # Merge con información de productos
        df = self.sell_in.merge(self.products, on='product_id', how='left')
        
        # Merge con stocks (opcional)
        df = df.merge(self.stocks, on=['periodo', 'product_id'], how='left')

        print("1. Creando features temporales...")
        df = self.create_time_features(df)
        
        print("2. Creando features de lag...")
        df = self.create_lag_features(df)
        
        print("3. Creando features de rolling...")
        df = self.create_rolling_features(df, max_window=12)
        
        print("4. Creando features de Prophet...")
        df = self.create_prophet_features(df)
        
        if fit_mode:
            print("5. Creando clusters DTW...")
            df = self.create_dtw_clusters(df)
        else:
            print("5. Asignando clusters DTW desde el modelo entrenado...")
            if hasattr(self, 'product_cluster_map'):
                # Mapear el cluster a cada producto
                df['dtw_cluster_50'] = df['product_id'].map(self.product_cluster_map)
                # Asignar cluster por defecto (-1) a productos nuevos no vistos en el entrenamiento
                df['dtw_cluster_50'] = df['dtw_cluster_50'].fillna(-1).astype(int)
            else:
                # Si por alguna razón no hay mapa de clusters, asignar default
                print("Advertencia: No se encontró mapa de clusters. Asignando -1.")
                df['dtw_cluster_50'] = -1 
                
        # print("6. Agregando features exógenas...")
        # df = self.add_external_features(df)
        
        print("7. Creando target ratio...")
        df = self.create_ratio_target(df)
        
        print("8. Manejando valores faltantes...")
        df = self.handle_missing_values(df)
        
        # print("9. Normalizando datos...")
        df = self.normalize_data(df, fit_scalers=fit_mode)
        
        print("10. Codificando variables categóricas...")
        df = self.encode_categorical_features(df, fit_encoders=fit_mode)
        
        # Filtrar columnas para el modelo
        feature_cols = [col for col in df.columns if col not in [
            'periodo', 'customer_id', 'product_id', # <-- FIX APLICADO
            'cust_request_qty', 'cust_request_tn',
            'tn', 'descripcion', 'tn_next_2', 'target_ratio' # Excluir el target también
        ]]
        
        if fit_mode:
            self.feature_names = feature_cols
        
        print(f"Feature engineering completado. Features: {len(self.feature_names)}")
        
        # Devolver el DataFrame completo y los nombres de las features
        return df, self.feature_names
        
    def optimize_hyperparameters(self, X_train, y_train, sample_weights, n_trials=10):
        """Optimización de hiperparámetros con Optuna"""
        if not OPTUNA_AVAILABLE:
            print("Optuna no disponible, usando parámetros predeterminados")
            return {
                'max_depth': 8, 'learning_rate': 0.1, 'n_estimators': 500,
                'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 0.1
            }
            
        print("Iniciando optimización de hiperparámetros...")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'max_bin': 1024, 'random_state': 42, 'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**params)
            
            split_idx = int(len(X_train) * 0.8)
            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]
            X_tr = X_train.iloc[:split_idx]
            y_tr = y_train.iloc[:split_idx]
            w_tr = sample_weights[:split_idx]
            
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            pred = model.predict(X_val)
            mse = mean_squared_error(y_val, pred)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Mejores parámetros: {study.best_params}")
        return study.best_params 
    
    def train_model(self, processed_df, feature_names, optimize=True):
        """Entrenar el modelo XGBoost"""
        print("Iniciando entrenamiento del modelo...")
        
        df = processed_df
        self.feature_names = feature_names
        
        train_mask = df['periodo'] < '2019-12-01'
        X_train = df[train_mask][self.feature_names]
        y_train = df[train_mask]['target_ratio']
        
        print(f"Datos de entrenamiento: {len(X_train)} muestras")
        
        sample_weights = self.calculate_sample_weights(df[train_mask])
        
        if optimize:
            best_params = self.optimize_hyperparameters(X_train, y_train, sample_weights)
        else:
            best_params = {
                'max_depth': 8, 'learning_rate': 0.1, 'n_estimators': 50,
                'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 0.1, 'max_bin': 1024, 'random_state': 42, 'n_jobs': -1
            }
        
        print("Entrenando modelo final...")
        self.xgb_model = xgb.XGBRegressor(**best_params)
        
        self.xgb_model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_train, y_train)],
            verbose=True
        )
        
        print("Entrenamiento completado!")      
              
    def predict_with_seeds(self, X, n_seeds=5):
        """Predicción con múltiples semillas"""
        predictions = []
        
        for seed in range(n_seeds):
            # Crear modelo con nueva semilla
            params = self.xgb_model.get_params()
            params['random_state'] = seed
            
            model_seed = xgb.XGBRegressor(**params)
            model_seed.load_model(self.xgb_model.save_model())
            
            pred = model_seed.predict(X)
            predictions.append(pred)
        
        # Promedio de predicciones
        return np.mean(predictions, axis=0)
        
    def predict(self, target_periods=['2020-02']):
        """Realizar predicciones para los productos objetivo"""
        print("Iniciando predicciones...")
        
        # Crear datos para predicción
        prediction_data = []
        last_period = pd.to_datetime('2019-12-01')
        
        for product_id in self.predict_products['product_id']:
            historical = self.sell_in[self.sell_in['product_id'] == product_id].copy()
            
            if len(historical) == 0:
                for period in target_periods:
                    prediction_data.append({
                        'periodo': pd.to_datetime(period), 'product_id': product_id,
                        'customer_id': 0, 'tn': 0, 'plan_precios_cuidados': 0,
                        'cust_request_qty': 0, 'cust_request_tn': 0
                    })
            else:
                last_data = historical[historical['periodo'] == last_period]
                if len(last_data) == 0:
                    last_data = historical[historical['periodo'] == historical['periodo'].max()]
                
                for period in target_periods:
                    for _, row in last_data.iterrows():
                        pred_row = row.copy()
                        pred_row['periodo'] = pd.to_datetime(period)
                        pred_row['tn'] = 0
                        prediction_data.append(pred_row.to_dict())
        
        pred_df = pd.DataFrame(prediction_data)
        
        print("Preparando features para predicción...")
        
        full_data = pd.concat([self.sell_in, pred_df], ignore_index=True)
        
        original_sell_in = self.sell_in.copy()
        self.sell_in = full_data
        
        pred_features_df, _ = self.prepare_features(fit_mode=True)
        
        pred_mask = pred_features_df['periodo'].isin([pd.to_datetime(p) for p in target_periods])
        X_pred = pred_features_df[pred_mask][self.feature_names]
        
        print(f"Realizando predicciones para {len(X_pred)} muestras...")
        
        ratio_predictions = self.xgb_model.predict(X_pred)
        
        print("Predicciones obtenidas, reconstruyendo resultados...")
        # Reconstruir predicciones en toneladas
        pred_results = pred_features_df[pred_mask][['periodo', 'product_id', 'customer_id']].copy()
        pred_results['predicted_ratio'] = ratio_predictions
        
        # Obtener último valor real para reconstruir
        last_values = {}
        for product_id in self.predict_products['product_id']:
            last_tn_series = original_sell_in[
                (original_sell_in['product_id'] == product_id) & 
                (original_sell_in['periodo'] == last_period)
            ]['tn']
            
            last_tn = last_tn_series.sum()
            
            if last_tn == 0 and not last_tn_series.empty:
                pass
            elif last_tn == 0:
                product_data = original_sell_in[original_sell_in['product_id'] == product_id]
                if not product_data.empty:
                    last_tn = product_data.sort_values('periodo').iloc[-1]['tn']
                else:
                    last_tn = 0.1
            
            last_values[product_id] = last_tn
        
        pred_results['last_tn'] = pred_results['product_id'].map(last_values)
        pred_results['predicted_tn'] = pred_results['predicted_ratio'] * pred_results['last_tn']
        
        # print("Desnormalizando predicciones...")
        # # Desnormalizar usando scalers por producto
        # for product_id in pred_results['product_id'].unique():
        #     if product_id in self.scalers:
        #         mask = pred_results['product_id'] == product_id
        #         scaler = self.scalers[product_id]
                
        #         scaled_values = pred_results.loc[mask, 'predicted_tn'].values.reshape(-1, 1)
        #         dummy_array = np.zeros((len(scaled_values), len(scaler.scale_)))
        #         dummy_array[:, 0] = scaled_values.flatten()
                
        #         try:
        #             denormalized = scaler.inverse_transform(dummy_array)
        #             pred_results.loc[mask, 'predicted_tn'] = denormalized[:, 0]
        #         except Exception:
        #             print(f"Error al desnormalizar para producto {product_id}. Usando predicción sin desnormalizar.")
        #             pass
        
        # Agrupar por producto y periodo
        final_predictions = pred_results.groupby(['product_id', 'periodo'])['predicted_tn'].sum().reset_index()
        
        # Restaurar sell_in original
        self.sell_in = original_sell_in
        
        print("Predicciones completadas!")
        return final_predictions    
    def save_model(self, path='sales_predictor_model.pkl'):
            """Guardar modelo y componentes"""
            model_dict = {
                'xgb_model': self.xgb_model,
                # 'scalers': self.scalers,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'cluster_model': self.cluster_model
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_dict, f)
            
            print(f"Modelo guardado en {path}")
            
    def load_model(self, path='sales_predictor_model.pkl'):
        """Cargar modelo y componentes"""
        with open(path, 'rb') as f:
            model_dict = pickle.load(f)
        
        self.xgb_model = model_dict['xgb_model']
        # self.scalers = model_dict['scalers']
        self.label_encoders = model_dict['label_encoders']
        self.feature_names = model_dict['feature_names']
        self.cluster_model = model_dict.get('cluster_model')
        
        print(f"Modelo cargado desde {path}")

def main():
    """Función principal para ejecutar el pipeline completo"""
    
    # Inicializar predictor
    predictor = SalesPredictor()
    
    # Cargar datos (ajustar rutas según tu estructura)
    predictor.load_data(
        sell_in_path='./../data/sell-in.txt',
        products_path='./../data/tb_productos.txt', 
        stocks_path='./../data/tb_stocks.txt',
        predict_products_path='./../data/product_id_apredecir201912.txt'
    )
    
    # --- LÓGICA DE CHECKPOINT ---
    checkpoint_path = 'features_procesadas.parquet'
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Cargando features desde checkpoint: {checkpoint_path}")
        processed_df = pd.read_parquet(checkpoint_path)
        
        # Regenerar la lista de features desde las columnas del DF cargado
        feature_names = [col for col in processed_df.columns if col not in [
            'periodo', 'customer_id', 'product_id',
            'cust_request_qty', 'cust_request_tn',
            'tn', 'descripcion', 'tn_next_2', 'target_ratio'
        ]]
        predictor.feature_names = feature_names # Asegurarse que el predictor la tenga
    else:
        print(f"⏳ No se encontró checkpoint. Generando features desde cero...")
        processed_df, feature_names = predictor.prepare_features(fit_mode=True)
        
        print("Revisando tipos de datos antes de guardar:")
        print(processed_df.info()) 
        
        print(f"💾 Guardando features en checkpoint: {checkpoint_path}")
        processed_df.to_parquet(checkpoint_path, index=False)

    # Entrenar modelo
    print("\n=== ENTRENAMIENTO ===")
    predictor.train_model(processed_df, feature_names, optimize=True) 
    # Realizar predicciones
    print("\n=== PREDICCIÓN ===")
    predictions = predictor.predict(['2020-02'])
    
    # Mostrar resultados
    print("\nPredicciones finales:")
    print(predictions.head(20))
    
    # Guardar modelo
    predictor.save_model('sales_model_complete.pkl')
    
    # Guardar predicciones
    predictions.to_csv('predicciones_finales.csv', index=False)
    print("Predicciones guardadas en predicciones_finales.csv")
    
    return predictions

# Ejecutar si es necesario
if __name__ == "__main__":
    predictions = main()