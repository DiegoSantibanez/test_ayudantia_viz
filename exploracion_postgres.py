import streamlit as st
import pandas as pd
import psycopg2
import sqlalchemy
from datetime import datetime
import plotly.express as px
import time

def show_exploracion_postgres():
    """Interfaz de exploración para PostgreSQL"""
    st.header("Explorador de Base de Datos PostgreSQL")
    st.write("Explora y consulta tu base de datos PostgreSQL de manera interactiva")
    
    # Inicializar session state
    if 'pg_connection_params' not in st.session_state:
        st.session_state.pg_connection_params = {
            'user': 'daso',
            'password': 'cleo1234',
            'host': 'localhost',
            'port': '5432',
            'database': 'postgres'
        }
    
    if 'pg_connected' not in st.session_state:
        st.session_state.pg_connected = False
    
    if 'pg_schemas' not in st.session_state:
        st.session_state.pg_schemas = []
    
    if 'pg_tables' not in st.session_state:
        st.session_state.pg_tables = {}
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    # Sidebar para configuración de conexión
    with st.sidebar:
        st.header("Configuración de Conexión")
        
        st.session_state.pg_connection_params['host'] = st.text_input(
            "Host", 
            value=st.session_state.pg_connection_params['host']
        )
        
        st.session_state.pg_connection_params['port'] = st.text_input(
            "Puerto", 
            value=st.session_state.pg_connection_params['port']
        )
        
        st.session_state.pg_connection_params['database'] = st.text_input(
            "Base de datos", 
            value=st.session_state.pg_connection_params['database']
        )
        
        st.session_state.pg_connection_params['user'] = st.text_input(
            "Usuario", 
            value=st.session_state.pg_connection_params['user']
        )
        
        st.session_state.pg_connection_params['password'] = st.text_input(
            "Contraseña", 
            type="password",
            value=st.session_state.pg_connection_params['password']
        )
        
        if st.button("Conectar", type="primary"):
            success, message = test_pg_connection(st.session_state.pg_connection_params)
            if success:
                st.success("Conexión exitosa")
                st.session_state.pg_connected = True
                # Cargar metadatos
                load_database_metadata()
                st.rerun()
            else:
                st.error(f"{message}")
                st.session_state.pg_connected = False
        
        if st.session_state.pg_connected:
            st.success("Conectado")
            
            # Información de la base de datos
            st.markdown("---")
            st.markdown("### Información de BD")
            if st.session_state.pg_schemas:
                st.write(f"**Esquemas:** {len(st.session_state.pg_schemas)}")
                total_tables = sum(len(tables) for tables in st.session_state.pg_tables.values())
                st.write(f"**Tablas:** {total_tables}")
        else:
            st.warning("Desconectado")
    
    # Contenido principal
    if st.session_state.pg_connected:
        # Pestañas principales
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Explorador", 
            "Consultas SQL", 
            "Análisis", 
            "Metadatos",
            "Dashboard"
        ])
        
        with tab1:
            show_database_explorer()
        
        with tab2:
            show_sql_query_interface()
        
        with tab3:
            show_data_analysis()
        
        with tab4:
            show_metadata_viewer()
        
        with tab5:
            show_database_dashboard()
    
    else:
        st.info("Por favor, configura y conecta a tu base de datos PostgreSQL en la barra lateral")

def test_pg_connection(params):
    """Prueba la conexión a PostgreSQL"""
    try:
        conn = psycopg2.connect(
            host=params['host'],
            port=params['port'],
            database=params['database'],
            user=params['user'],
            password=params['password']
        )
        conn.close()
        return True, "Conexión exitosa"
    except Exception as e:
        return False, f"Error de conexión: {str(e)}"

def get_pg_connection():
    """Obtiene una conexión SQLAlchemy"""
    params = st.session_state.pg_connection_params
    try:
        engine = sqlalchemy.create_engine(
            f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"
        )
        return engine
    except Exception as e:
        st.error(f"Error creando conexión: {str(e)}")
        return None

def load_database_metadata():
    """Carga metadatos de la base de datos"""
    try:
        engine = get_pg_connection()
        if engine is None:
            return
        
        # Obtener esquemas
        schemas_query = """
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
        ORDER BY schema_name;
        """
        schemas_df = pd.read_sql(schemas_query, engine)
        st.session_state.pg_schemas = schemas_df['schema_name'].tolist()
        
        # Obtener tablas para cada esquema
        st.session_state.pg_tables = {}
        for schema in st.session_state.pg_schemas:
            tables_query = f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = '{schema}' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """
            tables_df = pd.read_sql(tables_query, engine)
            st.session_state.pg_tables[schema] = tables_df['table_name'].tolist()
        
        engine.dispose()
    except Exception as e:
        st.error(f"Error cargando metadatos: {str(e)}")

def show_database_explorer():
    """Interfaz del explorador de base de datos"""
    st.subheader("Explorador de Base de Datos")
    
    if not st.session_state.pg_schemas:
        st.info("No se encontraron esquemas o no se han cargado los metadatos.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Navegación")
        
        # Selector de esquema
        selected_schema = st.selectbox(
            "Seleccionar esquema:",
            st.session_state.pg_schemas,
            key="explorer_schema"
        )
        
        if selected_schema and selected_schema in st.session_state.pg_tables:
            # Selector de tabla
            tables = st.session_state.pg_tables[selected_schema]
            if tables:
                selected_table = st.selectbox(
                    "Seleccionar tabla:",
                    tables,
                    key="explorer_table"
                )
                
                if selected_table:
                    # Información rápida de la tabla
                    try:
                        engine = get_pg_connection()
                        if engine:
                            # Contar filas
                            count_query = f'SELECT COUNT(*) as total FROM "{selected_schema}"."{selected_table}"'
                            count_result = pd.read_sql(count_query, engine)
                            total_rows = count_result['total'].iloc[0]
                            
                            # Información de columnas
                            columns_query = f"""
                            SELECT column_name, data_type, is_nullable
                            FROM information_schema.columns 
                            WHERE table_schema = '{selected_schema}' 
                            AND table_name = '{selected_table}'
                            ORDER BY ordinal_position;
                            """
                            columns_df = pd.read_sql(columns_query, engine)
                            
                            st.markdown("#### Información")
                            st.metric("Total de filas", f"{total_rows:,}")
                            st.metric("Columnas", len(columns_df))
                            
                            # Botones de acción
                            st.markdown("#### Acciones")
                            if st.button("Ver datos", key="view_data_btn"):
                                st.session_state.action = "view_data"
                            
                            if st.button("Estadísticas", key="stats_btn"):
                                st.session_state.action = "show_stats"
                            
                            if st.button("Describir tabla", key="describe_btn"):
                                st.session_state.action = "describe_table"
                            
                            engine.dispose()
                    except Exception as e:
                        st.error(f"Error obteniendo información: {str(e)}")
            else:
                st.info("No hay tablas en este esquema")
    
    with col2:
        st.markdown("### Contenido")
        
        if 'selected_schema' in locals() and 'selected_table' in locals():
            if hasattr(st.session_state, 'action'):
                if st.session_state.action == "view_data":
                    show_table_data(selected_schema, selected_table)
                elif st.session_state.action == "show_stats":
                    show_table_statistics(selected_schema, selected_table)
                elif st.session_state.action == "describe_table":
                    show_table_description(selected_schema, selected_table)
        else:
            st.info("Selecciona un esquema y tabla para ver su contenido")

def show_table_data(schema, table):
    """Muestra datos de la tabla"""
    try:
        engine = get_pg_connection()
        if engine is None:
            return
        
        st.markdown(f"#### Datos de {schema}.{table}")
        
        # Controles
        col1, col2, col3 = st.columns(3)
        with col1:
            limit = st.number_input("Límite de filas:", min_value=10, max_value=10000, value=100)
        with col2:
            offset = st.number_input("Offset:", min_value=0, value=0)
        with col3:
            if st.button("Actualizar"):
                st.rerun()
        
        # Consulta
        query = f'SELECT * FROM "{schema}"."{table}" LIMIT {limit} OFFSET {offset}'
        df = pd.read_sql(query, engine)
        
        st.dataframe(df, use_container_width=True)
        
        # Información adicional
        st.caption(f"Mostrando {len(df)} filas (Offset: {offset})")
        
        engine.dispose()
    except Exception as e:
        st.error(f"Error mostrando datos: {str(e)}")

def show_table_statistics(schema, table):
    """Muestra estadísticas de la tabla"""
    try:
        engine = get_pg_connection()
        if engine is None:
            return
        
        st.markdown(f"#### Estadísticas de {schema}.{table}")
        
        # Obtener columnas numéricas
        numeric_cols_query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        AND data_type IN ('integer', 'bigint', 'decimal', 'numeric', 'real', 'double precision', 'smallint')
        """
        numeric_cols_df = pd.read_sql(numeric_cols_query, engine)
        
        if len(numeric_cols_df) > 0:
            # Construir consulta de estadísticas
            stats_selects = []
            for _, row in numeric_cols_df.iterrows():
                col_name = row['column_name']
                stats_selects.extend([
                    f'AVG("{col_name}") as avg_{col_name}',
                    f'MIN("{col_name}") as min_{col_name}',
                    f'MAX("{col_name}") as max_{col_name}',
                    f'STDDEV("{col_name}") as std_{col_name}'
                ])
            
            stats_query = f'SELECT {", ".join(stats_selects)} FROM "{schema}"."{table}"'
            stats_df = pd.read_sql(stats_query, engine)
            
            # Mostrar estadísticas en formato amigable
            cols = st.columns(min(len(numeric_cols_df), 4))
            for i, (_, col_info) in enumerate(numeric_cols_df.iterrows()):
                col_name = col_info['column_name']
                with cols[i % len(cols)]:
                    st.markdown(f"**{col_name}**")
                    st.metric("Promedio", f"{stats_df[f'avg_{col_name}'].iloc[0]:.2f}")
                    st.metric("Mínimo", f"{stats_df[f'min_{col_name}'].iloc[0]:.2f}")
                    st.metric("Máximo", f"{stats_df[f'max_{col_name}'].iloc[0]:.2f}")
                    if stats_df[f'std_{col_name}'].iloc[0] is not None:
                        st.metric("Desv. Est.", f"{stats_df[f'std_{col_name}'].iloc[0]:.2f}")
        else:
            st.info("No se encontraron columnas numéricas para calcular estadísticas")
        
        # Contar valores únicos por columna
        st.markdown("#### Valores únicos por columna")
        
        all_cols_query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        ORDER BY ordinal_position
        """
        all_cols_df = pd.read_sql(all_cols_query, engine)
        
        unique_counts = []
        for col_name in all_cols_df['column_name']:
            count_query = f'SELECT COUNT(DISTINCT "{col_name}") as unique_count FROM "{schema}"."{table}"'
            count_result = pd.read_sql(count_query, engine)
            unique_counts.append({
                'Columna': col_name,
                'Valores únicos': count_result['unique_count'].iloc[0]
            })
        
        unique_df = pd.DataFrame(unique_counts)
        st.dataframe(unique_df, use_container_width=True, hide_index=True)
        
        engine.dispose()
    except Exception as e:
        st.error(f"Error calculando estadísticas: {str(e)}")

def show_table_description(schema, table):
    """Muestra descripción detallada de la tabla"""
    try:
        engine = get_pg_connection()
        if engine is None:
            return
        
        st.markdown(f"#### Descripción de {schema}.{table}")
        
        # Información de columnas
        columns_query = f"""
        SELECT 
            column_name as "Columna",
            data_type as "Tipo",
            is_nullable as "Permite NULL",
            column_default as "Valor por defecto",
            character_maximum_length as "Longitud máxima"
        FROM information_schema.columns 
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        ORDER BY ordinal_position;
        """
        columns_df = pd.read_sql(columns_query, engine)
        
        st.markdown("##### Estructura de columnas")
        st.dataframe(columns_df, use_container_width=True, hide_index=True)
        
        # Índices
        indexes_query = f"""
        SELECT 
            indexname as "Nombre del índice",
            indexdef as "Definición"
        FROM pg_indexes 
        WHERE schemaname = '{schema}' AND tablename = '{table}';
        """
        indexes_df = pd.read_sql(indexes_query, engine)
        
        if len(indexes_df) > 0:
            st.markdown("##### Índices")
            st.dataframe(indexes_df, use_container_width=True, hide_index=True)
        else:
            st.info("No se encontraron índices")
        
        # Tamaño de la tabla
        size_query = f"""
        SELECT pg_size_pretty(pg_total_relation_size('"{schema}"."{table}"')) as table_size;
        """
        size_df = pd.read_sql(size_query, engine)
        
        st.markdown("##### Información de almacenamiento")
        st.metric("Tamaño de la tabla", size_df['table_size'].iloc[0])
        
        engine.dispose()
    except Exception as e:
        st.error(f"Error obteniendo descripción: {str(e)}")

def show_sql_query_interface():
    """Interfaz para ejecutar consultas SQL"""
    st.subheader("Editor de Consultas SQL")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Editor SQL")
        
        # Consultas predefinidas
        predefined_queries = {
            "Seleccionar todo": "SELECT * FROM schema.tabla LIMIT 100;",
            "Contar registros": "SELECT COUNT(*) FROM schema.tabla;",
            "Describir tabla": """SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_schema = 'schema' AND table_name = 'tabla';""",
            "Top 10 por columna": "SELECT columna, COUNT(*) as frecuencia FROM schema.tabla GROUP BY columna ORDER BY frecuencia DESC LIMIT 10;"
        }
        
        selected_template = st.selectbox(
            "Plantillas de consulta:",
            ["Consulta personalizada"] + list(predefined_queries.keys())
        )
        
        # Editor de SQL
        if selected_template == "Consulta personalizada":
            initial_query = "-- Escribe tu consulta SQL aquí\nSELECT * FROM schema.tabla LIMIT 10;"
        else:
            initial_query = predefined_queries[selected_template]
        
        sql_query = st.text_area(
            "Consulta SQL:",
            value=initial_query,
            height=200,
            help="Escribe tu consulta SQL. Usa el formato schema.tabla para referenciar tablas."
        )
        
        # Botones de ejecución
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            execute_query = st.button("Ejecutar", type="primary")
        
        with col_b:
            explain_query = st.button("EXPLAIN")
        
        with col_c:
            save_query = st.button("Guardar")
    
    with col2:
        st.markdown("### Ayuda")
        
        # Autocompletado de esquemas y tablas
        if st.session_state.pg_schemas:
            with st.expander("Esquemas y Tablas"):
                for schema in st.session_state.pg_schemas:
                    st.write(f"**{schema}**")
                    if schema in st.session_state.pg_tables:
                        for table in st.session_state.pg_tables[schema][:5]:  # Mostrar máximo 5
                            st.write(f"  • {table}")
                        if len(st.session_state.pg_tables[schema]) > 5:
                            st.write(f"  ... y {len(st.session_state.pg_tables[schema]) - 5} más")
        
        # Historial de consultas
        with st.expander("Historial"):
            if st.session_state.query_history:
                for i, query in enumerate(reversed(st.session_state.query_history[-10:])):
                    if st.button(f"Query {len(st.session_state.query_history) - i}", key=f"history_{i}"):
                        sql_query = query
                        st.rerun()
                    st.caption(query[:50] + "..." if len(query) > 50 else query)
            else:
                st.info("No hay consultas en el historial")
    
    # Ejecutar consulta
    if execute_query and sql_query.strip():
        execute_sql_query(sql_query)
    
    if explain_query and sql_query.strip():
        execute_explain_query(sql_query)
    
    if save_query and sql_query.strip():
        st.session_state.query_history.append(sql_query.strip())
        st.success("Consulta guardada en el historial")

def execute_sql_query(query):
    """Ejecuta una consulta SQL"""
    try:
        engine = get_pg_connection()
        if engine is None:
            return
        
        start_time = time.time()
        
        # Verificar si es una consulta SELECT
        query_upper = query.upper().strip()
        if not query_upper.startswith('SELECT') and not query_upper.startswith('WITH'):
            st.warning("Solo se permiten consultas SELECT por seguridad")
            return
        
        # Ejecutar consulta
        df = pd.read_sql(query, engine)
        end_time = time.time()
        
        # Mostrar resultados
        st.markdown("### Resultados")
        
        # Métricas de la consulta
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas devueltas", len(df))
        with col2:
            st.metric("Columnas", len(df.columns))
        with col3:
            st.metric("Tiempo de ejecución", f"{end_time - start_time:.2f}s")
        
        # Mostrar datos
        if len(df) > 0:
            st.dataframe(df, use_container_width=True)
            
            # Opción de descarga
            csv = df.to_csv(index=False)
            st.download_button(
                label="Descargar resultados (CSV)",
                data=csv,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("La consulta no devolvió resultados")
        
        # Agregar al historial
        if query.strip() not in st.session_state.query_history:
            st.session_state.query_history.append(query.strip())
        
        engine.dispose()
    except Exception as e:
        st.error(f"Error ejecutando consulta: {str(e)}")

def execute_explain_query(query):
    """Ejecuta EXPLAIN para una consulta"""
    try:
        engine = get_pg_connection()
        if engine is None:
            return
        
        explain_query = f"EXPLAIN ANALYZE {query}"
        df = pd.read_sql(explain_query, engine)
        
        st.markdown("### Plan de Ejecución")
        st.text("\n".join(df.iloc[:, 0].tolist()))
        
        engine.dispose()
    except Exception as e:
        st.error(f"Error en EXPLAIN: {str(e)}")

def show_data_analysis():
    """Interfaz de análisis de datos"""
    st.subheader("Análisis de Datos")
    
    st.markdown("### Análisis Rápido")
    
    # Selector de tabla para análisis
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.pg_schemas:
            analysis_schema = st.selectbox(
                "Esquema para análisis:",
                st.session_state.pg_schemas,
                key="analysis_schema"
            )
    
    with col2:
        if 'analysis_schema' in locals() and analysis_schema in st.session_state.pg_tables:
            analysis_table = st.selectbox(
                "Tabla para análisis:",
                st.session_state.pg_tables[analysis_schema],
                key="analysis_table"
            )
    
    if 'analysis_schema' in locals() and 'analysis_table' in locals():
        try:
            engine = get_pg_connection()
            if engine is None:
                return
            
            # Obtener muestra de datos
            sample_query = f'SELECT * FROM "{analysis_schema}"."{analysis_table}" LIMIT 1000'
            df = pd.read_sql(sample_query, engine)
            
            if len(df) > 0:
                # Análisis básico
                st.markdown("#### Análisis Básico")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Filas analizadas", len(df))
                with col2:
                    st.metric("Columnas", len(df.columns))
                with col3:
                    st.metric("Valores nulos", df.isnull().sum().sum())
                with col4:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    st.metric("Columnas numéricas", len(numeric_cols))
                
                # Análisis por columnas
                st.markdown("#### Análisis por Columnas")
                
                # Selectores para análisis específico
                col1, col2 = st.columns(2)
                
                with col1:
                    analysis_type = st.selectbox(
                        "Tipo de análisis:",
                        ["Distribución", "Valores únicos", "Estadísticas descriptivas", "Correlaciones"]
                    )
                
                with col2:
                    if analysis_type in ["Distribución", "Valores únicos"]:
                        selected_column = st.selectbox(
                            "Columna a analizar:",
                            df.columns
                        )
                
                # Ejecutar análisis
                if analysis_type == "Distribución" and 'selected_column' in locals():
                    show_column_distribution(df, selected_column)
                
                elif analysis_type == "Valores únicos" and 'selected_column' in locals():
                    show_unique_values(df, selected_column)
                
                elif analysis_type == "Estadísticas descriptivas":
                    show_descriptive_stats(df)
                
                elif analysis_type == "Correlaciones":
                    show_correlations(df)
            
            engine.dispose()
        except Exception as e:
            st.error(f"Error en análisis: {str(e)}")

def show_column_distribution(df, column):
    """Muestra la distribución de una columna"""
    st.markdown(f"#### Distribución de {column}")
    
    if df[column].dtype in ['int64', 'float64']:
        # Histograma para columnas numéricas
        fig = px.histogram(df, x=column, title=f"Distribución de {column}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Gráfico de barras para columnas categóricas
        value_counts = df[column].value_counts().head(20)
        fig = px.bar(x=value_counts.index, y=value_counts.values,
                    title=f"Top 20 valores de {column}",
                    labels={'x': column, 'y': 'Frecuencia'})
        st.plotly_chart(fig, use_container_width=True)

def show_unique_values(df, column):
    """Muestra valores únicos de una columna"""
    st.markdown(f"#### Valores únicos de {column}")
    
    unique_values = df[column].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total valores únicos", len(unique_values))
    with col2:
        st.metric("Valor más común", str(unique_values.index[0]) if len(unique_values) > 0 else "N/A")
    
    # Mostrar tabla de frecuencias
    freq_df = pd.DataFrame({
        'Valor': unique_values.index,
        'Frecuencia': unique_values.values,
        'Porcentaje': (unique_values.values / len(df) * 100).round(2)
    })
    
    st.dataframe(freq_df.head(50), use_container_width=True, hide_index=True)

def show_descriptive_stats(df):
    """Muestra estadísticas descriptivas"""
    st.markdown("#### Estadísticas Descriptivas")
    
    numeric_df = df.select_dtypes(include=['number'])
    if len(numeric_df.columns) > 0:
        st.dataframe(numeric_df.describe(), use_container_width=True)
    else:
        st.info("No hay columnas numéricas para mostrar estadísticas")

def show_correlations(df):
    """Muestra matriz de correlaciones"""
    st.markdown("#### Matriz de Correlaciones")
    
    numeric_df = df.select_dtypes(include=['number'])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix,
                       text_auto=True,
                       aspect="auto",
                       title="Matriz de Correlaciones")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Se necesitan al menos 2 columnas numéricas para calcular correlaciones")

def show_metadata_viewer():
    """Visor de metadatos de la base de datos"""
    st.subheader("Metadatos de la Base de Datos")
    
    if not st.session_state.pg_schemas:
        st.info("No se han cargado metadatos. Asegúrate de estar conectado.")
        return
    
    # Resumen general
    st.markdown("### Resumen General")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Esquemas", len(st.session_state.pg_schemas))
    
    with col2:
        total_tables = sum(len(tables) for tables in st.session_state.pg_tables.values())
        st.metric("Tablas totales", total_tables)
    
    with col3:
        if st.button("Actualizar metadatos"):
            load_database_metadata()
            st.success("Metadatos actualizados")
            st.rerun()
    
    # Detalles por esquema
    st.markdown("### Detalles por Esquema")
    
    for schema in st.session_state.pg_schemas:
        with st.expander(f"Esquema: {schema}"):
            if schema in st.session_state.pg_tables:
                tables = st.session_state.pg_tables[schema]
                
                if tables:
                    st.write(f"**Tablas ({len(tables)}):**")
                    
                    # Mostrar tablas en columnas
                    cols = st.columns(3)
                    for i, table in enumerate(tables):
                        with cols[i % 3]:
                            if st.button(f"{table}", key=f"meta_{schema}_{table}"):
                                # Mostrar detalles de la tabla
                                show_table_metadata(schema, table)
                else:
                    st.info("No hay tablas en este esquema")

def show_table_metadata(schema, table):
    """Muestra metadatos detallados de una tabla"""
    try:
        engine = get_pg_connection()
        if engine is None:
            return
        
        st.markdown(f"#### Metadatos de {schema}.{table}")
        
        # Información de columnas con más detalles
        detailed_columns_query = f"""
        SELECT 
            column_name as "Columna",
            data_type as "Tipo",
            is_nullable as "NULL",
            column_default as "Por defecto",
            character_maximum_length as "Long. máx"
        FROM information_schema.columns 
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        ORDER BY ordinal_position;
        """
        
        columns_df = pd.read_sql(detailed_columns_query, engine)
        st.dataframe(columns_df, use_container_width=True, hide_index=True)
        
        engine.dispose()
    except Exception as e:
        st.error(f"Error obteniendo metadatos: {str(e)}")

def show_database_dashboard():
    """Dashboard con métricas de la base de datos"""
    st.subheader("Dashboard de la Base de Datos")
    
    try:
        engine = get_pg_connection()
        if engine is None:
            return
        
        # Métricas generales
        st.markdown("### Métricas Generales")
        
        # Información de conexiones
        connections_query = """
        SELECT count(*) as active_connections 
        FROM pg_stat_activity 
        WHERE state = 'active';
        """
        
        # Tamaño de la base de datos
        db_size_query = f"""
        SELECT pg_size_pretty(pg_database_size('{st.session_state.pg_connection_params["database"]}')) as db_size;
        """
        
        try:
            connections_df = pd.read_sql(connections_query, engine)
            db_size_df = pd.read_sql(db_size_query, engine)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Conexiones activas", connections_df['active_connections'].iloc[0])
            
            with col2:
                st.metric("Tamaño de BD", db_size_df['db_size'].iloc[0])
            
            with col3:
                st.metric("Esquemas", len(st.session_state.pg_schemas))
        except:
            st.info("Información de métricas no disponible (permisos limitados)")
        
        # Top tablas por tamaño
        st.markdown("### Top Tablas por Tamaño")
        
        try:
            top_tables_query = """
            SELECT 
                schemaname as "Esquema",
                tablename as "Tabla",
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as "Tamaño"
            FROM pg_tables 
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC 
            LIMIT 10;
            """
            
            top_tables_df = pd.read_sql(top_tables_query, engine)
            st.dataframe(top_tables_df, use_container_width=True, hide_index=True)
        except:
            st.info("Información de tamaños no disponible")
        
        # Actividad reciente
        st.markdown("### Actividad Reciente")
        
        try:
            activity_query = """
            SELECT 
                datname as "Base de datos",
                usename as "Usuario", 
                application_name as "Aplicación",
                state as "Estado",
                query_start as "Inicio consulta"
            FROM pg_stat_activity 
            WHERE state != 'idle' 
            ORDER BY query_start DESC 
            LIMIT 10;
            """
            
            activity_df = pd.read_sql(activity_query, engine)
            if len(activity_df) > 0:
                st.dataframe(activity_df, use_container_width=True, hide_index=True)
            else:
                st.info("No hay actividad reciente visible")
        except:
            st.info("Información de actividad no disponible (permisos limitados)")
        
        engine.dispose()
    except Exception as e:
        st.error(f"Error generando dashboard: {str(e)}")