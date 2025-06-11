import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import re
import json

# Funciones auxiliares para generar datos de ejemplo
@st.cache_data
def generate_sample_data():
    """Genera datos de ejemplo para las demos"""
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    data = {
        'fecha': dates,
        'ventas': np.random.randint(100, 1000, 365),
        'categoria': np.random.choice(['A', 'B', 'C'], 365),
        'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 365)
    }
    return pd.DataFrame(data)

@st.cache_data
def load_sample_csv():
    """Simula la carga de un archivo CSV"""
    return generate_sample_data()


def show_manejo_datos():
    """Sección de manejo de datos"""
    st.header("Manejo de Datos")
    st.write("Carga, manipulación y visualización de datos")
    
    # Pestañas para organizar el contenido
    tab1, tab2, tab3, tab4 = st.tabs(["Carga de Datos", "Exploración", "Edición", "Análisis"])
    
    with tab1:
        st.subheader("Carga de Archivos")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Subir archivo")
            uploaded_file = st.file_uploader(
                "Sube tu archivo de datos:",
                type=['csv', 'xlsx', 'json'],
                help="Formatos soportados: CSV, Excel, JSON"
            )
            
            # Opciones de configuración para CSV
            if uploaded_file and uploaded_file.name.endswith('.csv'):
                with st.expander("Configuración CSV"):
                    separator = st.selectbox("Separador:", [",", ";", "\t", "|", "^"], index=0)
                    encoding = st.selectbox("Codificación:", ["utf-8", "latin1", "cp1252"], index=0)
                    has_header = st.checkbox("Archivo tiene encabezados", value=True)
        
        with col2:
            st.markdown("### Datos de ejemplo")
            st.write("O puedes usar datos generados automáticamente:")
            
            if st.button("Generar datos de ejemplo"):
                st.session_state.sample_data = generate_sample_data()
                st.success("¡Datos de ejemplo generados!")
            
            # Opciones para los datos de ejemplo
            with st.expander("Configurar datos de ejemplo"):
                num_rows = st.slider("Número de filas:", 50, 1000, 365)
                include_nulls = st.checkbox("Incluir valores nulos", value=False)
                date_range = st.date_input(
                    "Rango de fechas:",
                    value=[datetime(2023, 1, 1), datetime(2023, 12, 31)],
                    format="DD/MM/YYYY"
                )
        
        # Procesar archivo subido
        df = None
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(
                        uploaded_file, 
                        sep=separator,
                        encoding=encoding,
                        header=0 if has_header else None
                    )
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.success(f"Archivo '{uploaded_file.name}' cargado exitosamente!")
                st.session_state.df = df
                
            except Exception as e:
                st.error(f"Error al cargar el archivo: {str(e)}")
        
        # Usar datos de ejemplo si están disponibles
        elif 'sample_data' in st.session_state:
            df = st.session_state.sample_data
            st.session_state.df = df
        
        # Mostrar código de carga
        with st.expander("Ver código de carga"):
            st.code('''
# Carga de archivo CSV
uploaded_file = st.file_uploader("Archivo:", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

# Datos de ejemplo
@st.cache_data
def generate_sample_data():
    dates = pd.date_range('2023-01-01', periods=365)
    return pd.DataFrame({
        'fecha': dates,
        'ventas': np.random.randint(100, 1000, 365),
        'categoria': np.random.choice(['A', 'B', 'C'], 365)
    })
            ''', language='python')
    
    with tab2:
        st.subheader("Exploración de Datos")
        
        if 'df' in st.session_state:
            df = st.session_state.df
            
            # Información general del dataset
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Filas", len(df))
            with col2:
                st.metric("Columnas", len(df.columns))
            with col3:
                st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                null_count = df.isnull().sum().sum()
                st.metric("Valores nulos", null_count)
            
            st.markdown("---")
            
            # Vista previa de los datos
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Vista previa")
                
                # Opciones de visualización
                view_option = st.radio(
                    "Selecciona vista:",
                    ["Primeras filas", "Últimas filas", "Muestra aleatoria"],
                    horizontal=True
                )
                
                num_rows_display = st.slider("Filas a mostrar:", 5, 50, 10)
                
                if view_option == "Primeras filas":
                    st.dataframe(df.head(num_rows_display), use_container_width=True)
                elif view_option == "Últimas filas":
                    st.dataframe(df.tail(num_rows_display), use_container_width=True)
                else:
                    st.dataframe(df.sample(num_rows_display), use_container_width=True)
            
            with col2:
                st.markdown("### Información de columnas")
                
                # Información detallada por columna
                column_info = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_count = df[col].isnull().sum()
                    unique_count = df[col].nunique()
                    
                    column_info.append({
                        'Columna': col,
                        'Tipo': dtype,
                        'Nulos': null_count,
                        'Únicos': unique_count
                    })
                
                info_df = pd.DataFrame(column_info)
                st.dataframe(info_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Estadísticas descriptivas
            st.markdown("### Estadísticas Descriptivas")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_cols = st.multiselect(
                    "Selecciona columnas numéricas:",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )
                
                if selected_cols:
                    st.dataframe(df[selected_cols].describe(), use_container_width=True)
            else:
                st.info("No hay columnas numéricas para mostrar estadísticas.")
            
            # Búsqueda y filtrado
            st.markdown("---")
            st.markdown("### Búsqueda y Filtrado")
            
            col1, col2 = st.columns(2)
            
            with col1:
                search_column = st.selectbox("Columna para buscar:", df.columns)
                search_term = st.text_input("Término de búsqueda:")
                
                if search_term:
                    mask = df[search_column].astype(str).str.contains(search_term, case=False, na=False)
                    filtered_df = df[mask]
                    st.write(f"Encontradas {len(filtered_df)} filas")
                    st.dataframe(filtered_df.head(10), use_container_width=True)
            
            with col2:
                # Filtro por rango (para columnas numéricas)
                if numeric_cols:
                    filter_column = st.selectbox("Columna numérica para filtrar:", numeric_cols)
                    min_val = float(df[filter_column].min())
                    max_val = float(df[filter_column].max())
                    
                    range_values = st.slider(
                        f"Rango para {filter_column}:",
                        min_val, max_val,
                        (min_val, max_val)
                    )
                    
                    filtered_by_range = df[
                        (df[filter_column] >= range_values[0]) & 
                        (df[filter_column] <= range_values[1])
                    ]
                    st.write(f"Filas en rango: {len(filtered_by_range)}")
                    st.dataframe(filtered_by_range.head(10), use_container_width=True)
        
        else:
            st.info("Primero carga un dataset en la pestaña 'Carga de Datos'")
        
        # Código de exploración
        with st.expander("Ver código de exploración"):
            st.code('''
# Información básica
st.metric("Filas", len(df))
st.metric("Columnas", len(df.columns))

# Vista previa
st.dataframe(df.head(10))

# Estadísticas descriptivas
st.dataframe(df.describe())

# Búsqueda
search_term = st.text_input("Buscar:")
if search_term:
    mask = df['columna'].str.contains(search_term, case=False)
    st.dataframe(df[mask])
            ''', language='python')
    
    with tab3:
        st.subheader("Edición de Datos")
        
        if 'df' in st.session_state:
            df = st.session_state.df.copy()
            
            st.markdown("### Operaciones de Limpieza")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Manejo de valores nulos")
                
                null_columns = df.columns[df.isnull().any()].tolist()
                if null_columns:
                    selected_null_col = st.selectbox("Columna con nulos:", null_columns)
                    null_action = st.selectbox(
                        "Acción para valores nulos:",
                        ["Eliminar filas", "Rellenar con media", "Rellenar con mediana", "Rellenar con valor personalizado"]
                    )
                    
                    if null_action == "Rellenar con valor personalizado":
                        fill_value = st.text_input("Valor para rellenar:")
                    
                    if st.button("Aplicar limpieza de nulos"):
                        if null_action == "Eliminar filas":
                            df = df.dropna(subset=[selected_null_col])
                        elif null_action == "Rellenar con media":
                            if df[selected_null_col].dtype in ['int64', 'float64']:
                                df[selected_null_col].fillna(df[selected_null_col].mean(), inplace=True)
                        elif null_action == "Rellenar con mediana":
                            if df[selected_null_col].dtype in ['int64', 'float64']:
                                df[selected_null_col].fillna(df[selected_null_col].median(), inplace=True)
                        elif null_action == "Rellenar con valor personalizado" and fill_value:
                            df[selected_null_col].fillna(fill_value, inplace=True)
                        
                        st.session_state.df = df
                        st.success("¡Limpieza aplicada!")
                        st.rerun()
                else:
                    st.info("No hay valores nulos en el dataset")
            
            with col2:
                st.markdown("#### Transformaciones de columnas")
                
                transform_col = st.selectbox("Columna a transformar:", df.columns)
                transform_action = st.selectbox(
                    "Tipo de transformación:",
                    ["Convertir a mayúsculas", "Convertir a minúsculas", "Eliminar espacios", "Convertir tipo de dato"]
                )
                
                if transform_action == "Convertir tipo de dato":
                    new_type = st.selectbox("Nuevo tipo:", ["string", "int", "float", "datetime"])
                
                if st.button("Aplicar transformación"):
                    try:
                        if transform_action == "Convertir a mayúsculas":
                            df[transform_col] = df[transform_col].astype(str).str.upper()
                        elif transform_action == "Convertir a minúsculas":
                            df[transform_col] = df[transform_col].astype(str).str.lower()
                        elif transform_action == "Eliminar espacios":
                            df[transform_col] = df[transform_col].astype(str).str.strip()
                        elif transform_action == "Convertir tipo de dato":
                            if new_type == "datetime":
                                df[transform_col] = pd.to_datetime(df[transform_col])
                            else:
                                df[transform_col] = df[transform_col].astype(new_type)
                        
                        st.session_state.df = df
                        st.success("¡Transformación aplicada!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error en transformación: {str(e)}")
            
            st.markdown("---")
            
            # Editor de datos interactivo
            st.markdown("### Editor Interactivo")
            st.write("Edita los datos directamente en la tabla:")
            
            edited_df = st.data_editor(
                df.head(20),  # Mostrar solo las primeras 20 filas para edición
                use_container_width=True,
                num_rows="dynamic"  # Permite agregar/eliminar filas
            )
            
            if st.button("Guardar cambios"):
                # En una aplicación real, aquí guardarías los cambios
                st.success("¡Cambios guardados! (En esta demo, los cambios son temporales)")
            
            # Agregar nueva columna
            st.markdown("---")
            st.markdown("### Agregar Nueva Columna")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_col_name = st.text_input("Nombre de la nueva columna:")
            with col2:
                new_col_type = st.selectbox("Tipo de columna:", ["Valor constante", "Cálculo basado en otras columnas"])
            with col3:
                if new_col_type == "Valor constante":
                    new_col_value = st.text_input("Valor:")
                else:
                    calculation_col = st.selectbox("Columna base:", df.select_dtypes(include=[np.number]).columns)
                    operation = st.selectbox("Operación:", ["* 2", "/ 2", "+ 100", "- 100"])
            
            if st.button("Agregar columna") and new_col_name:
                if new_col_type == "Valor constante":
                    df[new_col_name] = new_col_value
                else:
                    if operation == "* 2":
                        df[new_col_name] = df[calculation_col] * 2
                    elif operation == "/ 2":
                        df[new_col_name] = df[calculation_col] / 2
                    elif operation == "+ 100":
                        df[new_col_name] = df[calculation_col] + 100
                    elif operation == "- 100":
                        df[new_col_name] = df[calculation_col] - 100
                
                st.session_state.df = df
                st.success(f"¡Columna '{new_col_name}' agregada!")
                st.rerun()
        
        else:
            st.info("Primero carga un dataset en la pestaña 'Carga de Datos'")
        
        # Código de edición
        with st.expander("Ver código de edición"):
            st.code('''
# Editor interactivo
edited_df = st.data_editor(df, num_rows="dynamic")

# Limpieza de nulos
df_clean = df.dropna()  # o df.fillna(value)

# Transformaciones
df['columna'] = df['columna'].str.upper()
df['fecha'] = pd.to_datetime(df['fecha'])

# Agregar columna
df['nueva_columna'] = df['columna_existente'] * 2
            ''', language='python')
    
    with tab4:
        st.subheader("Análisis de Datos")
        
        if 'df' in st.session_state:
            df = st.session_state.df
            
            # Análisis univariado
            st.markdown("### Análisis Univariado")
            
            analysis_col = st.selectbox("Selecciona columna para analizar:", df.columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if df[analysis_col].dtype in ['int64', 'float64']:
                    # Análisis numérico
                    st.markdown(f"#### Estadísticas de {analysis_col}")
                    stats = df[analysis_col].describe()
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Media", f"{stats['mean']:.2f}")
                        st.metric("Mediana", f"{df[analysis_col].median():.2f}")
                    with col_b:
                        st.metric("Desv. Estándar", f"{stats['std']:.2f}")
                        st.metric("Rango", f"{stats['max'] - stats['min']:.2f}")
                    
                    # Histograma
                    fig = px.histogram(df, x=analysis_col, title=f"Distribución de {analysis_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Análisis categórico
                    st.markdown(f"#### Frecuencias de {analysis_col}")
                    value_counts = df[analysis_col].value_counts()
                    
                    st.dataframe(
                        pd.DataFrame({
                            'Valor': value_counts.index,
                            'Frecuencia': value_counts.values,
                            'Porcentaje': (value_counts.values / len(df) * 100).round(2)
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Gráfico de barras
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f"Frecuencias de {analysis_col}",
                               labels={'x': analysis_col, 'y': 'Frecuencia'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot para columnas numéricas
                if df[analysis_col].dtype in ['int64', 'float64']:
                    fig = px.box(df, y=analysis_col, title=f"Box Plot - {analysis_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Gráfico de pie para categóricas
                    value_counts = df[analysis_col].value_counts().head(10)  # Top 10
                    fig = px.pie(values=value_counts.values, names=value_counts.index,
                               title=f"Distribución de {analysis_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Análisis bivariado
            st.markdown("### Análisis Bivariado")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("Variable X:", df.columns, key="x_var")
            with col2:
                y_col = st.selectbox("Variable Y:", df.columns, key="y_var")
            
            if x_col != y_col:
                # Determinar tipo de gráfico según tipos de datos
                x_is_numeric = df[x_col].dtype in ['int64', 'float64']
                y_is_numeric = df[y_col].dtype in ['int64', 'float64']
                
                if x_is_numeric and y_is_numeric:
                    # Scatter plot
                    fig = px.scatter(df, x=x_col, y=y_col, 
                                   title=f"Relación entre {x_col} y {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlación
                    correlation = df[x_col].corr(df[y_col])
                    st.metric("Correlación", f"{correlation:.3f}")
                
                elif x_is_numeric or y_is_numeric:
                    # Box plot o violin plot
                    if x_is_numeric:
                        fig = px.box(df, x=y_col, y=x_col, 
                                   title=f"{x_col} por {y_col}")
                    else:
                        fig = px.box(df, x=x_col, y=y_col, 
                                   title=f"{y_col} por {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Tabla cruzada para categóricas
                    crosstab = pd.crosstab(df[x_col], df[y_col])
                    st.write("Tabla cruzada:")
                    st.dataframe(crosstab, use_container_width=True)
                    
                    # Heatmap
                    fig = px.imshow(crosstab.values, 
                                  x=crosstab.columns, 
                                  y=crosstab.index,
                                  title=f"Heatmap: {x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Resumen del dataset
            st.markdown("### Resumen del Dataset")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Columnas Numéricas")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    for col in numeric_cols:
                        st.write(f"• {col}")
                else:
                    st.write("No hay columnas numéricas")
            
            with col2:
                st.markdown("#### Columnas de Texto")
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                if text_cols:
                    for col in text_cols:
                        st.write(f"• {col}")
                else:
                    st.write("No hay columnas de texto")
            
            with col3:
                st.markdown("#### Columnas de Fecha")
                date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                if date_cols:
                    for col in date_cols:
                        st.write(f"• {col}")
                else:
                    st.write("No hay columnas de fecha")
            
            # Exportar datos procesados
            st.markdown("---")
            st.markdown("### Exportar Datos")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Descargar como CSV",
                    data=csv,
                    file_name="datos_procesados.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Para Excel necesitarías openpyxl
                st.info("Excel export requiere openpyxl")
            
            with col3:
                json_data = df.to_json(indent=2)
                st.download_button(
                    label="Descargar como JSON",
                    data=json_data,
                    file_name="datos_procesados.json",
                    mime="application/json"
                )
        
        else:
            st.info("Primero carga un dataset en la pestaña 'Carga de Datos'")
        
        # Código de análisis
        with st.expander("Ver código de análisis"):
            st.code('''
# Estadísticas descriptivas
st.dataframe(df.describe())

# Análisis univariado
fig = px.histogram(df, x='columna')
st.plotly_chart(fig)

# Análisis bivariado
fig = px.scatter(df, x='col1', y='col2')
st.plotly_chart(fig)

# Correlación
correlation = df['col1'].corr(df['col2'])
st.metric("Correlación", f"{correlation:.3f}")

# Exportar
csv = df.to_csv(index=False)
st.download_button("Descargar CSV", csv, "data.csv")
            ''', language='python')
