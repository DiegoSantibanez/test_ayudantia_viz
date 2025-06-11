import streamlit as st
import pandas as pd
import psycopg2
from psycopg2 import sql
import sqlalchemy
from datetime import datetime
import io
import base64

def show_subida_postgres():
    """Función principal para mostrar la sección de subida a PostgreSQL"""
    
    # Inicializar session state
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'connection_params' not in st.session_state:
        st.session_state.connection_params = {
            'user': 'daso',
            'password': 'cleo1234',
            'host': 'localhost',
            'port': '5432',
            'database': 'postgres',
            'schema': 'test_fiut'
        }
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'table_created' not in st.session_state:
        st.session_state.table_created = False

    # HEADER
    st.title("🐘 Gestor de Carga PostgreSQL")
    st.markdown("Aplicación para cargar archivos CSV/Excel a bases de datos PostgreSQL con VARCHAR dinámico")

    # SIDEBAR - Configuración de conexión
    with st.sidebar:
        st.header("⚙️ Configuración de Conexión")
        
        st.session_state.connection_params['user'] = st.text_input(
            "Usuario", 
            value=st.session_state.connection_params['user'],
            key="pg_user"
        )
        
        st.session_state.connection_params['password'] = st.text_input(
            "Contraseña", 
            value=st.session_state.connection_params['password'],
            type="password",
            key="pg_password"
        )
        
        st.session_state.connection_params['host'] = st.text_input(
            "Host", 
            value=st.session_state.connection_params['host'],
            key="pg_host"
        )
        
        st.session_state.connection_params['port'] = st.text_input(
            "Puerto", 
            value=st.session_state.connection_params['port'],
            key="pg_port"
        )
        
        st.session_state.connection_params['database'] = st.text_input(
            "Base de datos", 
            value=st.session_state.connection_params['database'],
            key="pg_database"
        )
        
        st.session_state.connection_params['schema'] = st.text_input(
            "Schema", 
            value=st.session_state.connection_params['schema'],
            key="pg_schema"
        )
        
        if st.button("🔌 Probar Conexión", key="test_connection"):
            success, message = test_connection(st.session_state.connection_params)
            if success:
                st.success(message)
                add_log(message, "success")
            else:
                st.error(message)
                add_log(message, "error")

    # MAIN PANEL
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Carga de Archivo", "⚙️ Configuración de Tabla", "📝 SQL Preview", "📊 Log de Operaciones"])

    # Tab 1: Carga de Archivo
    with tab1:
        st.header("📁 Carga y Exploración de Archivo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Seleccionar archivo CSV o Excel",
                type=['csv', 'xlsx', 'xls'],
                key="file_uploader_pg"
            )
            
            if uploaded_file is not None:
                file_type = uploaded_file.name.split('.')[-1].lower()
                
                st.subheader("Configuración de lectura")
                
                if file_type == 'csv':
                    separator = st.selectbox(
                        "Separador",
                        [',', ';', '\t', '^', '|', 'Personalizado'],
                        key="separator_select"
                    )
                    if separator == 'Personalizado':
                        separator = st.text_input("Ingrese separador personalizado", key="custom_separator")
                    
                    encoding = st.selectbox(
                        "Codificación",
                        ['UTF-8', 'Latin-1', 'ISO-8859-1', 'Personalizado'],
                        key="encoding_select"
                    )
                    if encoding == 'Personalizado':
                        encoding = st.text_input("Ingrese codificación personalizada", key="custom_encoding")
                    
                    skip_rows = st.number_input(
                        "Filas a omitir al inicio",
                        min_value=0,
                        value=0,
                        key="skip_rows"
                    )
                    
                    na_values = st.text_input(
                        "Valores a considerar como nulos (separados por coma)",
                        value="NA, N/A, null, NULL",
                        key="na_values"
                    )
        
        with col2:
            if uploaded_file is not None:
                if st.button("📖 Leer archivo", key="read_file_button"):
                    try:
                        if file_type == 'csv':
                            df_raw = pd.read_csv(
                                uploaded_file,
                                sep=separator,
                                encoding=encoding,
                                skiprows=skip_rows,
                                na_values=na_values.split(',') if na_values else None
                            )
                        else:
                            df_raw = pd.read_excel(
                                uploaded_file,
                                skiprows=skip_rows,
                                na_values=na_values.split(',') if na_values else None
                            )
                        
                        # Aplicar corrección de tipos de datos
                        st.session_state.df = fix_dataframe_dtypes(df_raw)
                        
                        add_log(f"Archivo '{uploaded_file.name}' cargado exitosamente", "success")
                        st.success(f"✅ Archivo cargado: {len(st.session_state.df)} filas, {len(st.session_state.df.columns)} columnas")
                        
                    except Exception as e:
                        st.error(f"Error al leer archivo: {str(e)}")
                        add_log(f"Error al leer archivo: {str(e)}", "error")
        
        # Vista previa de datos
        if st.session_state.df is not None:
            st.subheader("Vista previa de datos")
            
            # Estadísticas básicas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Filas", len(st.session_state.df))
            with col2:
                st.metric("Columnas", len(st.session_state.df.columns))
            with col3:
                st.metric("Valores nulos", st.session_state.df.isnull().sum().sum())
            with col4:
                st.metric("Memoria", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Mostrar primeras filas - con manejo de errores adicional
            try:
                st.dataframe(st.session_state.df.head(10))
            except Exception as e:
                st.warning(f"Error mostrando vista previa: {str(e)}")
                st.write("Información básica del DataFrame:")
                st.write(f"Forma: {st.session_state.df.shape}")
                st.write(f"Columnas: {list(st.session_state.df.columns)}")
            
            # Información de columnas
            with st.expander("📊 Información de columnas"):
                try:
                    col_info = pd.DataFrame({
                        'Columna': st.session_state.df.columns,
                        'Tipo': st.session_state.df.dtypes.astype(str),
                        'Valores no nulos': st.session_state.df.count(),
                        'Valores únicos': [st.session_state.df[col].nunique() for col in st.session_state.df.columns]
                    })
                    st.dataframe(col_info)
                except Exception as e:
                    st.warning(f"Error mostrando información de columnas: {str(e)}")

    # Tab 2: Configuración de Tabla
    with tab2:
        st.header("⚙️ Configuración de Tabla")
        
        if st.session_state.df is not None:
            col1, col2 = st.columns(2)
            
            # Inicializar column_mapping aquí para asegurar que esté disponible
            column_mapping = {}
            
            with col1:
                # Nombre de tabla
                table_name = st.text_input(
                    "Nombre de la tabla",
                    value="nueva_tabla",
                    key="table_name_input"
                )
                
                # Opciones de llave primaria
                st.subheader("🔑 Configuración de Llave Primaria")
                add_id_column = st.checkbox(
                    "Agregar columna ID autoincremental",
                    value=True,
                    key="add_id_checkbox"
                )
                
                # Renombrar columnas - mover antes de la configuración de PK
                st.subheader("✏️ Renombrar Columnas")
                rename_cols = st.checkbox("Habilitar renombrado de columnas", value=False, key="rename_checkbox")
                
                if rename_cols:
                    for col in st.session_state.df.columns:
                        new_name = st.text_input(
                            f"Renombrar '{col}' a:",
                            value=col,
                            key=f"rename_{col}_pg"
                        )
                        if new_name != col:
                            column_mapping[col] = new_name
                
                if not add_id_column:
                    # Usar nombres renombrados para la selección de PK
                    pk_options = []
                    for col in st.session_state.df.columns:
                        display_name = column_mapping.get(col, col)
                        pk_options.append(display_name)
                    
                    pk_columns = st.multiselect(
                        "Seleccionar columnas como llave primaria",
                        options=pk_options,
                        key="pk_columns_select"
                    )
                else:
                    pk_columns = []
                
                # Opción de fecha de carga
                add_fecha_carga = st.checkbox(
                    "Agregar columna fecha_carga (YYYYMMDD)",
                    value=True,
                    key="add_fecha_checkbox"
                )
            
            with col2:
                # Configuración de tipos de datos
                st.subheader("📋 Tipos de datos PostgreSQL")
                column_types = {}
                
                # Usar nombres de columnas renombradas si existen
                display_columns = []
                for col in st.session_state.df.columns:
                    display_name = column_mapping.get(col, col)
                    display_columns.append((col, display_name))
                
                for original_col, display_col in display_columns:
                    # Obtener tipo sugerido con análisis dinámico
                    suggested_type = get_postgresql_type(str(st.session_state.df[original_col].dtype), st.session_state.df[original_col])
                    
                    # Crear lista de opciones con el tipo sugerido primero
                    type_options = ['VARCHAR(255)', 'INTEGER', 'BIGINT', 'NUMERIC', 'BOOLEAN', 'DATE', 'TIMESTAMP', 'TEXT']
                    if suggested_type not in type_options:
                        type_options.insert(0, suggested_type)
                    
                    column_types[display_col] = st.selectbox(
                        f"{display_col}",
                        options=type_options,
                        index=0 if suggested_type not in ['VARCHAR(255)', 'INTEGER', 'BIGINT', 'NUMERIC', 'BOOLEAN', 'DATE', 'TIMESTAMP', 'TEXT'] else type_options.index(suggested_type),
                        key=f"type_{original_col}_pg",
                        help=f"Tipo sugerido: {suggested_type}" if suggested_type != type_options[0] else None
                    )
            
            # Opciones adicionales
            st.subheader("⚡ Opciones adicionales")
            col1, col2 = st.columns(2)
            with col1:
                create_if_not_exists = st.checkbox("CREATE TABLE IF NOT EXISTS", value=True, key="create_if_not_exists")
            with col2:
                drop_if_exists = st.checkbox("DROP TABLE IF EXISTS", value=False, key="drop_if_exists")
            
            # Información sobre rendimiento y VARCHAR dinámico
            st.info("💡 **Nota sobre VARCHAR dinámico**: El sistema analiza automáticamente las columnas de texto y sugiere tamaños de VARCHAR basados en la longitud máxima de los datos + 20% de buffer. Si el texto es muy largo (>10,000 caracteres), se sugiere usar TEXT.")
            st.info("💡 **Nota sobre rendimiento**: Los datos se insertan en lotes para optimizar el tiempo de carga. Para archivos grandes (>10,000 filas), el proceso mostrará el progreso real por lotes.")
        else:
            st.warning("⚠️ Por favor, carga un archivo primero en la pestaña 'Carga de Archivo'")

    # Tab 3: SQL Preview
    with tab3:
        st.header("📝 Vista Previa SQL y Ejecución")
        
        if st.session_state.df is not None and 'table_name' in locals():
            # Generar SQL
            create_sql = generate_create_table_sql(
                table_name,
                st.session_state.df,
                column_types,
                pk_columns,
                add_id_column,
                add_fecha_carga,
                st.session_state.connection_params['schema'],
                column_mapping
            )
            
            # Agregar modificadores
            if drop_if_exists:
                drop_sql = f'DROP TABLE IF EXISTS "{st.session_state.connection_params["schema"]}"."{table_name}";\n\n'
                create_sql = drop_sql + create_sql
            
            if create_if_not_exists and not drop_if_exists:
                create_sql = create_sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
            
            # Mostrar SQL
            st.subheader("Vista previa de SQL")
            st.code(create_sql, language='sql')
            
            # Botón de descarga
            st.download_button(
                label="📥 Descargar SQL",
                data=create_sql,
                file_name=f"{table_name}_create.sql",
                mime="text/plain",
                key="download_sql_button"
            )
            
            # Botones de ejecución
            st.subheader("🚀 Ejecución")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🏗️ Crear Tabla", type="primary", key="create_table_button"):
                    success, message = execute_sql(st.session_state.connection_params, create_sql)
                    if success:
                        st.success(message)
                        add_log(f"Tabla '{table_name}' creada exitosamente", "success")
                        st.session_state.table_created = True
                    else:
                        st.error(message)
                        add_log(message, "error")
            
            with col2:
                if st.button("📤 Insertar Datos", disabled=not st.session_state.table_created, key="insert_data_button"):
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(value):
                            progress_bar.progress(value)
                            status_text.text(f"Progreso: {int(value * 100)}%")
                        
                        with st.spinner("Insertando datos..."):
                            success, message = insert_data(
                                st.session_state.connection_params,
                                table_name,
                                st.session_state.df,
                                add_fecha_carga,
                                st.session_state.connection_params['schema'],
                                progress_callback=update_progress,
                                column_mapping=column_mapping
                            )
                        
                        if success:
                            st.success(message)
                            add_log(message, "success")
                        else:
                            st.error(message)
                            add_log(message, "error")
            
            if not st.session_state.table_created:
                st.info("ℹ️ Debes crear la tabla antes de insertar datos")
        else:
            st.warning("⚠️ Por favor, carga un archivo y configura la tabla primero")

    # Tab 4: Log de Operaciones
    with tab4:
        st.header("📊 Log de Operaciones")
        
        if st.session_state.log_messages:
            # Botón para limpiar log
            if st.button("🗑️ Limpiar Log", key="clear_log_button"):
                st.session_state.log_messages = []
                st.rerun()
            
            # Mostrar mensajes del log
            for msg in reversed(st.session_state.log_messages):
                if msg['type'] == 'success':
                    st.success(f"**{msg['timestamp']}** - {msg['message']}")
                elif msg['type'] == 'error':
                    st.error(f"**{msg['timestamp']}** - {msg['message']}")
                else:
                    st.info(f"**{msg['timestamp']}** - {msg['message']}")
        else:
            st.info("No hay operaciones registradas aún")

# Funciones auxiliares
def add_log(message, type="info"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_messages.append({
        'timestamp': timestamp,
        'message': message,
        'type': type
    })

def fix_dataframe_dtypes(df):
    """
    Convierte tipos de datos problemáticos para compatibilidad con PyArrow/Streamlit
    """
    df_fixed = df.copy()
    
    for col in df_fixed.columns:
        dtype = str(df_fixed[col].dtype)
        
        # Convertir tipos nullable de pandas a tipos estándar
        if 'Int64' in dtype:
            df_fixed[col] = df_fixed[col].astype('float64')  # Usar float64 para manejar NaN
        elif 'Int32' in dtype:
            df_fixed[col] = df_fixed[col].astype('float64')
        elif 'Int16' in dtype:
            df_fixed[col] = df_fixed[col].astype('float64')
        elif 'Int8' in dtype:
            df_fixed[col] = df_fixed[col].astype('float64')
        elif 'Float64' in dtype:
            df_fixed[col] = df_fixed[col].astype('float64')
        elif 'Float32' in dtype:
            df_fixed[col] = df_fixed[col].astype('float32')
        elif 'boolean' in dtype:
            df_fixed[col] = df_fixed[col].astype('object')  # Convertir boolean nullable a object
        elif 'string' in dtype:
            df_fixed[col] = df_fixed[col].astype('object')
    
    return df_fixed

def test_connection(params):
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

def calculate_optimal_varchar_length(series):
    """Calcula la longitud óptima de VARCHAR basada en los datos reales"""
    try:
        # Filtrar valores nulos y convertir a string
        non_null_values = series.dropna().astype(str)
        
        if len(non_null_values) == 0:
            return 255  # Valor por defecto si no hay datos
        
        # Calcular longitud máxima
        max_length = non_null_values.str.len().max()
        
        # Si la longitud máxima es menor o igual a 255, usar 255 como estándar
        if max_length <= 255:
            return 255
        
        # Calcular con 20% adicional
        optimal_length = int(max_length * 1.2)
        
        # Asegurar que no exceda límites razonables de PostgreSQL
        # VARCHAR puede ir hasta 65535, pero es mejor usar TEXT para valores muy grandes
        if optimal_length > 10000:
            return None  # Retornar None para indicar que se debe usar TEXT
        
        return optimal_length
        
    except Exception as e:
        add_log(f"Error calculando longitud VARCHAR para columna: {str(e)}", "error")
        return 255

def get_postgresql_type(dtype, series=None):
    """Mapea tipos de pandas a PostgreSQL con VARCHAR dinámico"""
    dtype_str = str(dtype).lower()
    
    # Manejar tipos nullable de pandas
    if 'int64' in dtype_str or 'int32' in dtype_str or 'int16' in dtype_str or 'int8' in dtype_str:
        return 'INTEGER'
    elif 'float64' in dtype_str or 'float32' in dtype_str:
        return 'NUMERIC'
    elif dtype_str == 'object' and series is not None:
        # Calcular longitud óptima para columnas de texto
        optimal_length = calculate_optimal_varchar_length(series)
        if optimal_length is None:
            return 'TEXT'
        else:
            return f'VARCHAR({optimal_length})'
    elif dtype_str == 'object':
        return 'VARCHAR(255)'  # Fallback si no se proporciona la serie
    elif 'datetime64' in dtype_str:
        return 'TIMESTAMP'
    elif 'bool' in dtype_str:
        return 'BOOLEAN'
    else:
        return 'VARCHAR(255)'

def generate_create_table_sql(table_name, df, column_types, pk_columns, add_id, add_fecha, schema, column_mapping=None):
    """Genera la sentencia CREATE TABLE"""
    columns = []
    
    # Agregar columna ID si se seleccionó
    if add_id:
        columns.append("id SERIAL PRIMARY KEY")
    
    # Agregar columnas del DataFrame
    for col in df.columns:
        # Usar nombre renombrado si existe
        display_name = column_mapping.get(col, col) if column_mapping else col
        col_type = column_types.get(display_name, get_postgresql_type(str(df[col].dtype), df[col]))
        col_def = f'"{display_name}" {col_type}'
        
        # Verificar si es llave primaria (usando el nombre renombrado)
        if display_name in pk_columns and not add_id:
            col_def += " PRIMARY KEY" if len(pk_columns) == 1 else ""
        columns.append(col_def)
    
    # Agregar columna fecha_carga si se seleccionó
    if add_fecha:
        columns.append("fecha_carga INTEGER")
    
    # Agregar constraint de llave primaria compuesta si es necesario
    if len(pk_columns) > 1 and not add_id:
        # CORRECIÓN: Separar la creación de la lista de columnas citadas
        quoted_columns = [f'"{col}"' for col in pk_columns]
        pk_constraint = f"PRIMARY KEY ({', '.join(quoted_columns)})"
        columns.append(pk_constraint)
    
    # Construir sentencia completa
    full_table_name = f'"{schema}"."{table_name}"'
    create_sql = f"CREATE TABLE {full_table_name} (\n    " + ",\n    ".join(columns) + "\n);"
    
    return create_sql

def execute_sql(params, sql_query):
    """Ejecuta una consulta SQL"""
    try:
        conn = psycopg2.connect(
            host=params['host'],
            port=params['port'],
            database=params['database'],
            user=params['user'],
            password=params['password']
        )
        cur = conn.cursor()
        cur.execute(sql_query)
        conn.commit()
        cur.close()
        conn.close()
        return True, "Operación exitosa"
    except Exception as e:
        return False, f"Error al ejecutar SQL: {str(e)}"

def insert_data(params, table_name, df, add_fecha, schema, progress_callback=None, column_mapping=None):
    """Inserta datos en la tabla con optimización por chunks"""
    try:
        # Crear conexión con SQLAlchemy
        engine = sqlalchemy.create_engine(
            f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}",
            connect_args={'options': '-csearch_path={}'.format(schema)}
        )
        
        # Preparar DataFrame
        df_to_insert = df.copy()
        
        # Convertir tipos problemáticos antes de la inserción
        for col in df_to_insert.columns:
            dtype_str = str(df_to_insert[col].dtype).lower()
            if 'int64' in dtype_str or 'int32' in dtype_str:
                # Convertir a int estándar, manejando NaN
                df_to_insert[col] = df_to_insert[col].astype('Int64').astype('object')
                df_to_insert[col] = df_to_insert[col].where(pd.notnull(df_to_insert[col]), None)
        
        # Renombrar columnas si se especificó
        if column_mapping:
            df_to_insert.rename(columns=column_mapping, inplace=True)
        
        if add_fecha:
            df_to_insert['fecha_carga'] = int(datetime.now().strftime('%Y%m%d'))
        
        # Calcular tamaño óptimo de chunk
        total_rows = len(df_to_insert)
        chunk_size = min(1000, max(100, total_rows // 10))  # Entre 100 y 1000 filas por chunk
        
        # Insertar datos por chunks
        chunks_processed = 0
        total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)
        
        for i in range(0, total_rows, chunk_size):
            chunk = df_to_insert.iloc[i:i+chunk_size]
            chunk.to_sql(
                table_name,
                engine,
                schema=schema,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=None  # Ya estamos manejando chunks manualmente
            )
            
            chunks_processed += 1
            if progress_callback:
                progress_callback(chunks_processed / total_chunks)
        
        engine.dispose()
        return True, f"Se insertaron {total_rows} registros exitosamente en {total_chunks} lotes"
    except Exception as e:
        return False, f"Error al insertar datos: {str(e)}"

def analyze_text_columns(df):
    """Analiza las columnas de texto y proporciona estadísticas de longitud"""
    text_analysis = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            non_null_values = df[col].dropna().astype(str)
            if len(non_null_values) > 0:
                lengths = non_null_values.str.len()
                text_analysis[col] = {
                    'max_length': lengths.max(),
                    'avg_length': lengths.mean(),
                    'recommended_varchar': calculate_optimal_varchar_length(df[col]),
                    'total_values': len(non_null_values)
                }
    
    return text_analysis