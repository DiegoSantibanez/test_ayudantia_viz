import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time


def show_layout_diseno():
    """Sección de layout y diseño"""
    st.header("Layout y Diseño")
    st.write("Diferentes opciones de diseño y organización")
    
    # Tabs principales para organizar los diferentes layouts
    layout_tabs = st.tabs([
        "Columnas", 
        "Tabs", 
        "Containers", 
        "Expanders", 
        "Formularios",
        "Combinaciones"
    ])
    
    with layout_tabs[0]:
        st.subheader("Sistema de Columnas")
        st.write("Las columnas permiten organizar el contenido horizontalmente")
        
        # Ejemplo básico de columnas
        st.markdown("### Columnas Básicas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Columna 1**")
            st.write("Este es el contenido de la primera columna")
            st.button("Botón 1", key="layout_btn1")
            
        with col2:
            st.markdown("**Columna 2**")
            st.write("Este es el contenido de la segunda columna")
            st.selectbox("Opciones", ["A", "B", "C"], key="layout_select1")
            
        with col3:
            st.markdown("**Columna 3**")
            st.write("Este es el contenido de la tercera columna")
            st.metric("Métrica", "123", "5%")
        
        # Código ejemplo
        with st.expander("Ver código de columnas básicas"):
            st.code('''
col1, col2, col3 = st.columns(3)

with col1:
    st.write("Contenido columna 1")
    
with col2:
    st.write("Contenido columna 2")
    
with col3:
    st.write("Contenido columna 3")
            ''', language='python')
        
        st.markdown("---")
        
        # Columnas con diferentes proporciones
        st.markdown("### Columnas con Proporciones Personalizadas")
        
        col_a, col_b, col_c = st.columns([3, 1, 2])
        
        with col_a:
            st.markdown("**Columna Ancha (3)**")
            st.write("Esta columna ocupa 3 unidades de espacio")
            st.slider("Slider ejemplo", 0, 100, 50, key="layout_slider1")
            
        with col_b:
            st.markdown("**Columna Estrecha (1)**")
            st.write("1 unidad")
            
        with col_c:
            st.markdown("**Columna Media (2)**")
            st.write("Esta columna ocupa 2 unidades")
        
        with st.expander("Ver código de columnas con proporciones"):
            st.code('''
# Proporciones personalizadas
col1, col2, col3 = st.columns([3, 1, 2])

# La primera columna es 3 veces más ancha que la segunda
# La tercera columna es 2 veces más ancha que la segunda
            ''', language='python')
        
        st.markdown("---")
        
        # Columnas con gaps
        st.markdown("### Columnas con Espaciado (Gaps)")
        
        col_x, space, col_y = st.columns([2, 0.5, 2])
        
        with col_x:
            st.markdown("**Columna Izquierda**")
            st.info("Espacio entre columnas usando una columna vacía")
            
        with col_y:
            st.markdown("**Columna Derecha**")
            st.success("El espacio se crea con una columna de proporción 0.5")
        
        with st.expander("Ver código de columnas con espaciado"):
            st.code('''
# Crear espacio entre columnas
col1, space, col2 = st.columns([2, 0.5, 2])

with col1:
    st.write("Contenido izquierda")
    
# No usar 'space' - queda vacío
    
with col2:
    st.write("Contenido derecha")
            ''', language='python')
        
        st.markdown("---")
        
        # Columnas anidadas
        st.markdown("### Columnas Anidadas")
        
        main_col1, main_col2 = st.columns(2)
        
        with main_col1:
            st.markdown("**Columna Principal 1**")
            
            # Sub-columnas dentro de la primera columna principal
            sub_col1, sub_col2 = st.columns(2)
            
            with sub_col1:
                st.write("Sub-columna 1.1")
                st.button("Botón A", key="layout_btn_a")
                
            with sub_col2:
                st.write("Sub-columna 1.2")
                st.button("Botón B", key="layout_btn_b")
                
        with main_col2:
            st.markdown("**Columna Principal 2**")
            st.write("Esta columna no tiene sub-columnas")
            st.text_area("Área de texto", height=100, key="layout_textarea1")
        
        with st.expander("Ver código de columnas anidadas"):
            st.code('''
# Columnas principales
main_col1, main_col2 = st.columns(2)

with main_col1:
    st.write("Columna principal 1")
    
    # Sub-columnas
    sub_col1, sub_col2 = st.columns(2)
    
    with sub_col1:
        st.write("Sub-columna 1.1")
    
    with sub_col2:
        st.write("Sub-columna 1.2")

with main_col2:
    st.write("Columna principal 2")
            ''', language='python')
    
    with layout_tabs[1]:
        st.subheader("Sistema de Tabs")
        st.write("Las tabs permiten organizar contenido en pestañas navegables")
        
        # Tabs básicas
        st.markdown("### Tabs Básicas")
        
        tab1, tab2, tab3 = st.tabs(["Datos", "Gráficos", "Configuración"])
        
        with tab1:
            st.write("### Contenido de la pestaña Datos")
            
            # Generar datos de ejemplo
            df_example = pd.DataFrame({
                'Producto': ['A', 'B', 'C', 'D'],
                'Ventas': [100, 150, 120, 200],
                'Stock': [50, 30, 45, 60]
            })
            
            st.dataframe(df_example, use_container_width=True)
            
        with tab2:
            st.write("### Contenido de la pestaña Gráficos")
            
            # Gráfico simple
            fig = px.bar(df_example, x='Producto', y='Ventas', title="Ventas por Producto")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.write("### Contenido de la pestaña Configuración")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.checkbox("Activar modo oscuro", key="layout_dark_mode")
                st.checkbox("Mostrar leyenda", key="layout_show_legend")
                
            with config_col2:
                st.radio("Tema", ["Claro", "Oscuro", "Automático"], key="layout_theme")
        
        with st.expander("Ver código de tabs básicas"):
            st.code('''
# Crear tabs
tab1, tab2, tab3 = st.tabs(["Datos", "Gráficos", "Configuración"])

with tab1:
    st.write("Contenido de Datos")
    st.dataframe(df)

with tab2:
    st.write("Contenido de Gráficos")
    st.plotly_chart(fig)

with tab3:
    st.write("Contenido de Configuración")
    st.checkbox("Opción 1")
            ''', language='python')
        
        st.markdown("---")
        
        # Tabs dinámicas
        st.markdown("### Tabs Dinámicas")
        
        # Número de tabs configurable
        num_tabs = st.slider("Número de pestañas:", 2, 6, 3, key="layout_num_tabs")
        
        # Crear tabs dinámicamente
        tab_names = [f"Tab {i+1}" for i in range(num_tabs)]
        dynamic_tabs = st.tabs(tab_names)
        
        for i, tab in enumerate(dynamic_tabs):
            with tab:
                st.write(f"### Contenido de Tab {i+1}")
                st.write(f"Esta es la pestaña número {i+1} de {num_tabs}")
                
                # Contenido diferente según el índice
                if i % 2 == 0:
                    st.info(f"Tab par: {i+1}")
                else:
                    st.warning(f"Tab impar: {i+1}")
        
        with st.expander("Ver código de tabs dinámicas"):
            st.code('''
# Número de tabs configurable
num_tabs = st.slider("Número de pestañas:", 2, 6, 3)

# Crear tabs dinámicamente
tab_names = [f"Tab {i+1}" for i in range(num_tabs)]
tabs = st.tabs(tab_names)

for i, tab in enumerate(tabs):
    with tab:
        st.write(f"Contenido de Tab {i+1}")
            ''', language='python')
        
        st.markdown("---")
        
        # Tabs anidadas
        st.markdown("### Tabs Anidadas")
        
        main_tab1, main_tab2 = st.tabs(["Principal 1", "Principal 2"])
        
        with main_tab1:
            st.write("### Tab Principal 1")
            
            # Sub-tabs dentro de la primera tab principal
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Sub-tab A", "Sub-tab B", "Sub-tab C"])
            
            with sub_tab1:
                st.write("Contenido de Sub-tab A")
                st.text_input("Input en sub-tab A", key="layout_input_a")
                
            with sub_tab2:
                st.write("Contenido de Sub-tab B")
                st.slider("Slider en sub-tab B", 0, 10, 5, key="layout_slider_b")
                
            with sub_tab3:
                st.write("Contenido de Sub-tab C")
                st.button("Botón en sub-tab C", key="layout_btn_c")
                
        with main_tab2:
            st.write("### Tab Principal 2")
            st.write("Esta tab no tiene sub-tabs")
            st.selectbox("Selección", ["Opción 1", "Opción 2", "Opción 3"], key="layout_select_main2")

        with st.expander("Ver código de tabs dinámicas"):
            st.code('''
        main_tab1, main_tab2 = st.tabs(["Principal 1", "Principal 2"])
        
        with main_tab1:
            st.write("### Tab Principal 1")
            
            # Sub-tabs dentro de la primera tab principal
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Sub-tab A", "Sub-tab B", "Sub-tab C"])
            
            with sub_tab1:
                st.write("Contenido de Sub-tab A")
                st.text_input("Input en sub-tab A", key="layout_input_a")
                
            with sub_tab2:
                st.write("Contenido de Sub-tab B")
                st.slider("Slider en sub-tab B", 0, 10, 5, key="layout_slider_b")
                
            with sub_tab3:
                st.write("Contenido de Sub-tab C")
                st.button("Botón en sub-tab C", key="layout_btn_c")
                
        with main_tab2:
            st.write("### Tab Principal 2")
            st.write("Esta tab no tiene sub-tabs")
            st.selectbox("Selección", ["Opción 1", "Opción 2", "Opción 3"], key="layout_select_main2")
            ''', language='python')
    
    with layout_tabs[2]:
        st.subheader("Containers")
        st.write("Los containers permiten agrupar y organizar elementos")
        
        # Container básico
        st.markdown("### Container Básico")
        
        with st.container():
            st.write("Este contenido está dentro de un container")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Métrica 1", "100", "10%")
            with col2:
                st.metric("Métrica 2", "200", "-5%")
                
            st.info("Los containers son útiles para agrupar elementos relacionados")
        
        with st.expander("Ver código de container básico"):
            st.code('''
with st.container():
    st.write("Contenido del container")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Métrica 1", "100")
    with col2:
        st.metric("Métrica 2", "200")
            ''', language='python')
        
        st.markdown("---")
        
        # Container con bordes (usando CSS)
        st.markdown("### Container con Estilo Personalizado")
        
        # CSS para container con borde
        st.markdown("""
        <style>
        .custom-container {
            border: 2px solid #1f77b4;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        
        container1 = st.container()
        with container1:
            st.markdown("#### Container con Bordes y Estilo")
            st.write("Este container tiene estilos CSS personalizados")
            
            progress = st.progress(0)
            for i in range(100):
                progress.progress(i + 1)
                time.sleep(0.01)
            
            st.success("¡Proceso completado!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Empty containers
        st.markdown("### Empty Containers (Placeholders)")
        
        st.write("Los empty containers permiten actualizar contenido dinámicamente")
        
        # Crear placeholders
        placeholder1 = st.empty()
        placeholder2 = st.empty()
        
        # Botones para actualizar contenido
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Mostrar Texto", key="layout_show_text"):
                placeholder1.text("¡Texto actualizado!")
                placeholder2.info("Información adicional")
                
        with col2:
            if st.button("Mostrar Métrica", key="layout_show_metric"):
                placeholder1.metric("Valor", "42", "↑12%")
                placeholder2.success("Métrica actualizada")
                
        with col3:
            if st.button("Limpiar", key="layout_clear"):
                placeholder1.empty()
                placeholder2.empty()
        
        with st.expander("Ver código de empty containers"):
            st.code('''
# Crear placeholders
placeholder = st.empty()

# Actualizar contenido
if st.button("Actualizar"):
    placeholder.text("Nuevo contenido")
    
# Limpiar
if st.button("Limpiar"):
    placeholder.empty()
            ''', language='python')
        
        st.markdown("---")
        
        # Sidebar container
        st.markdown("### Sidebar Container")
        
        st.write("El sidebar es un container especial para navegación y controles")
        
        with st.sidebar:
            st.markdown("### Controles del Layout")
            
            layout_option = st.selectbox(
                "Tipo de layout:",
                ["Simple", "Complejo", "Dashboard"],
                key="layout_sidebar_option"
            )
            
            st.slider("Ancho de columnas:", 1, 12, 6, key="layout_sidebar_width")
            
            st.markdown("---")
            
            with st.expander("Opciones avanzadas"):
                st.checkbox("Modo debug", key="layout_debug")
                st.color_picker("Color tema:", "#1f77b4", key="layout_color")
    
    with layout_tabs[3]:
        st.subheader("Expanders")
        st.write("Los expanders permiten ocultar/mostrar contenido")
        
        # Expander básico
        st.markdown("### Expander Básico")
        
        with st.expander("Click para expandir"):
            st.write("Este contenido está oculto por defecto")
            st.write("Puedes agregar cualquier elemento aquí:")
            
            # Elementos dentro del expander
            name = st.text_input("Nombre:", key="layout_expander_name")
            age = st.number_input("Edad:", min_value=0, max_value=120, key="layout_expander_age")
            
            if st.button("Enviar", key="layout_expander_submit"):
                st.success(f"Datos recibidos: {name}, {age} años")
        
        with st.expander("Ver código de expander básico"):
            st.code('''
with st.expander("Click para expandir"):
    st.write("Contenido oculto")
    
    # Cualquier elemento de Streamlit
    name = st.text_input("Nombre:")
    if st.button("Enviar"):
        st.success(f"Hola {name}")
            ''', language='python')
        
        st.markdown("---")
        
        # Expander expandido por defecto
        st.markdown("### Expander Expandido por Defecto")
        
        with st.expander("Este expander está abierto", expanded=True):
            st.write("El parámetro `expanded=True` lo mantiene abierto inicialmente")
            
            # Gráfico dentro del expander
            data = pd.DataFrame({
                'x': range(10),
                'y': np.random.randn(10).cumsum()
            })
            
            fig = px.line(data, x='x', y='y', title="Gráfico en Expander")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Múltiples expanders
        st.markdown("### Múltiples Expanders (Acordeón)")
        
        expander_data = {
            "Sección de Datos": {
                "content": "Información sobre datos y estadísticas",
                "metrics": {"Usuarios": 1234, "Ventas": 5678, "Productos": 890}
            },
            "Sección de Análisis": {
                "content": "Herramientas y métodos de análisis",
                "metrics": {"Precisión": "95%", "Recall": "92%", "F1-Score": "93.5%"}
            },
            "Sección de Configuración": {
                "content": "Ajustes y parámetros del sistema",
                "metrics": {"CPU": "45%", "RAM": "2.3GB", "Disco": "120GB"}
            }
        }
        
        for title, data in expander_data.items():
            with st.expander(title):
                st.write(data["content"])
                
                # Mostrar métricas en columnas
                cols = st.columns(len(data["metrics"]))
                for i, (metric, value) in enumerate(data["metrics"].items()):
                    with cols[i]:
                        st.metric(metric, value)
        
        st.markdown("---")
        
        # Expanders anidados
        st.markdown("### Expanders con Contenido Complejo")
        
        with st.expander("Dashboard Compacto"):
            # Tabs dentro de expander
            tab1, tab2, tab3 = st.tabs(["Resumen", "Detalles", "Configuración"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("KPI 1", "100", "↑10%")
                with col2:
                    st.metric("KPI 2", "250", "↓5%")
                with col3:
                    st.metric("KPI 3", "75", "→0%")
                    
            with tab2:
                st.write("Detalles del análisis...")
                st.progress(75)
                
            with tab3:
                st.checkbox("Activar notificaciones", key="layout_notif")
                st.slider("Umbral de alerta:", 0, 100, 50, key="layout_threshold")
    
    with layout_tabs[4]:
        st.subheader("Formularios")
        st.write("Los formularios agrupan inputs y tienen un botón de envío único")
        
        # Formulario básico
        st.markdown("### Formulario Básico")
        
        with st.form("formulario_basico"):
            st.write("#### Registro de Usuario")
            
            # Inputs del formulario
            nombre = st.text_input("Nombre completo:")
            email = st.text_input("Email:")
            edad = st.number_input("Edad:", min_value=18, max_value=100, value=25)
            
            col1, col2 = st.columns(2)
            with col1:
                pais = st.selectbox("País:", ["España", "México", "Argentina", "Colombia"])
            with col2:
                genero = st.radio("Género:", ["Masculino", "Femenino", "Otro"])
            
            acepta_terminos = st.checkbox("Acepto los términos y condiciones")
            
            # Botón de envío (obligatorio en formularios)
            submitted = st.form_submit_button("Registrar")
            
            if submitted:
                if acepta_terminos:
                    st.success(f"Usuario {nombre} registrado correctamente!")
                    st.write(f"Email: {email}")
                    st.write(f"País: {pais}")
                else:
                    st.error("Debes aceptar los términos y condiciones")
        
        with st.expander("Ver código de formulario básico"):
            st.code('''
with st.form("mi_formulario"):
    st.write("### Formulario de Registro")
    
    nombre = st.text_input("Nombre:")
    email = st.text_input("Email:")
    edad = st.number_input("Edad:", min_value=18)
    
    # Importante: form_submit_button en lugar de button
    submitted = st.form_submit_button("Enviar")
    
    if submitted:
        st.success(f"Registrado: {nombre}")
            ''', language='python')
        
        st.markdown("---")
        
        # Formulario con validación
        st.markdown("### Formulario con Validación")
        
        with st.form("formulario_validacion"):
            st.write("#### Formulario de Contacto")
            
            col1, col2 = st.columns(2)
            
            with col1:
                nombre = st.text_input("Nombre:", help="Mínimo 3 caracteres")
                telefono = st.text_input("Teléfono:", help="Formato: +56 X XXXX XXXX")
                
            with col2:
                email = st.text_input("Email:", help="ejemplo@dominio.com")
                asunto = st.selectbox("Asunto:", 
                    ["Consulta General", "Soporte Técnico", "Ventas", "Otro"])
            
            mensaje = st.text_area("Mensaje:", height=100, help="Mínimo 10 caracteres")
            
            urgente = st.checkbox("Marcar como urgente")
            
            # Columnas para los botones
            col1, col2, col3 = st.columns([1, 1, 3])
            
            with col1:
                submitted = st.form_submit_button("Enviar", type="primary")
            with col2:
                clear = st.form_submit_button("Limpiar")
            
            if submitted:
                # Validaciones
                errors = []
                
                if len(nombre) < 3:
                    errors.append("El nombre debe tener al menos 3 caracteres")
                
                if "@" not in email or "." not in email:
                    errors.append("Email inválido")
                
                if len(mensaje) < 10:
                    errors.append("El mensaje debe tener al menos 10 caracteres")
                
                if errors:
                    for error in errors:
                        st.error(f"{error}")
                else:
                    st.success("Formulario enviado correctamente!")
                    st.balloons()
            
            if clear:
                st.info("Formulario limpiado (recarga la página)")
        
        st.markdown("---")
        
        # Formulario horizontal
        st.markdown("### Formulario Horizontal")
        
        with st.form("formulario_horizontal"):
            cols = st.columns([2, 2, 1, 1])
            
            with cols[0]:
                search_term = st.text_input("Buscar:", label_visibility="collapsed", 
                                          placeholder="Término de búsqueda...")
            with cols[1]:
                category = st.selectbox("Categoría:", 
                    ["Todas", "Productos", "Servicios", "Blog"],
                    label_visibility="collapsed")
            with cols[2]:
                date_filter = st.date_input("Fecha:", label_visibility="collapsed")
            with cols[3]:
                search_button = st.form_submit_button("Buscar", use_container_width=True)
            
            if search_button:
                st.info(f"Buscando '{search_term}' en {category} para {date_filter}")
        
        st.markdown("---")
        
        # Formulario con archivos
        st.markdown("### Formulario con Carga de Archivos")
        
        with st.form("formulario_archivos"):
            st.write("#### Subir Documentación")
            
            col1, col2 = st.columns(2)
            
            with col1:
                doc_type = st.selectbox("Tipo de documento:",
                    ["Factura", "Contrato", "Informe", "Otro"])
                doc_date = st.date_input("Fecha del documento:")
                
            with col2:
                uploaded_file = st.file_uploader("Seleccionar archivo:", 
                    type=['pdf', 'doc', 'docx', 'txt'])
                notes = st.text_area("Notas adicionales:", height=76)
            
            col1, col2 = st.columns([1, 5])
            
            with col1:
                submitted = st.form_submit_button("Subir", use_container_width=True)
                
            if submitted:
                if uploaded_file is not None:
                    st.success(f"Archivo '{uploaded_file.name}' subido correctamente")
                    st.write(f"Tipo: {doc_type}")
                    st.write(f"Tamaño: {uploaded_file.size / 1024:.2f} KB")
                else:
                    st.error("Por favor selecciona un archivo")
    
    with layout_tabs[5]:
        st.subheader("Combinaciones de Layouts")
        st.write("Ejemplos de cómo combinar diferentes elementos de layout")
        
        # Dashboard complejo
        st.markdown("### Dashboard Complejo")
        
        # Header con métricas
        metrics_container = st.container()
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Usuarios Activos", "1,234", "↑12%")
            with col2:
                st.metric("Ingresos", "$45,678", "↑8%")
            with col3:
                st.metric("Conversión", "3.4%", "↓0.2%")
            with col4:
                st.metric("Satisfacción", "4.5/5", "↑0.3")
        
        st.markdown("---")
        
        # Layout principal con sidebar simulado
        main_col, sidebar_col = st.columns([3, 1])
        
        with sidebar_col:
            st.markdown("#### Filtros")
            
            with st.expander("Período", expanded=True):
                period = st.radio("Seleccionar:", 
                    ["Hoy", "Semana", "Mes", "Año"], 
                    key="layout_period")
                
            with st.expander("Categorías", expanded=True):
                categories = st.multiselect("Incluir:",
                    ["Ventas", "Marketing", "Soporte", "Desarrollo"],
                    default=["Ventas", "Marketing"],
                    key="layout_categories")
                    
            with st.expander("Opciones", expanded=False):
                show_trend = st.checkbox("Mostrar tendencia", True, key="layout_trend")
                show_comparison = st.checkbox("Comparar períodos", False, key="layout_compare")
        
        with main_col:
            # Tabs dentro de la columna principal
            tab1, tab2, tab3 = st.tabs(["Vista General", "Análisis", "Datos"])
            
            with tab1:
                # Sub-layout con gráficos
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Gráfico de ejemplo
                    data = pd.DataFrame({
                        'fecha': pd.date_range('2024-01-01', periods=30),
                        'valor': np.random.randn(30).cumsum() + 100
                    })
                    fig = px.line(data, x='fecha', y='valor', 
                                title="Evolución Temporal")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with chart_col2:
                    # Otro gráfico
                    pie_data = pd.DataFrame({
                        'categoria': ['A', 'B', 'C', 'D'],
                        'valor': [30, 25, 20, 25]
                    })
                    fig2 = px.pie(pie_data, values='valor', names='categoria',
                                 title="Distribución por Categoría")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Expander con detalles
                with st.expander("Ver detalles del análisis"):
                    st.write("Información adicional sobre los gráficos...")
                    
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    with detail_col1:
                        st.metric("Min", "85.2")
                    with detail_col2:
                        st.metric("Max", "125.7")
                    with detail_col3:
                        st.metric("Promedio", "103.4")
                        
            with tab2:
                st.write("### Análisis Detallado")
                
                # Formulario de análisis dentro de tab
                with st.form("analysis_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        analysis_type = st.selectbox("Tipo de análisis:",
                            ["Tendencia", "Correlación", "Distribución"])
                    with col2:
                        confidence = st.slider("Nivel de confianza:", 
                            0.9, 0.99, 0.95, 0.01)
                    
                    run_analysis = st.form_submit_button("Ejecutar Análisis")
                    
                    if run_analysis:
                        with st.spinner("Analizando datos..."):
                            time.sleep(2)
                        st.success("Análisis completado!")
                        
            with tab3:
                st.write("### Tabla de Datos")
                
                # Generar datos de ejemplo
                table_data = pd.DataFrame({
                    'ID': range(1, 11),
                    'Producto': [f'Producto {i}' for i in range(1, 11)],
                    'Ventas': np.random.randint(100, 1000, 10),
                    'Stock': np.random.randint(0, 100, 10),
                    'Estado': np.random.choice(['Activo', 'Inactivo'], 10)
                })
                
                st.dataframe(table_data, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Layout tipo Card
        st.markdown("### Layout tipo Cards")
        
        # CSS para cards
        st.markdown("""
        <style>
        .card {
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
            margin-bottom: 1rem;
            background-color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        card_cols = st.columns(3)
        
        for i, col in enumerate(card_cols):
            with col:
                with st.container():
                    st.markdown(f"#### Card {i+1}")
                    st.write(f"Contenido de la tarjeta {i+1}")
                    
                    if i == 0:
                        st.progress(0.7)
                        st.caption("70% completado")
                    elif i == 1:
                        st.metric("Valor", f"{np.random.randint(50, 200)}", 
                                f"{np.random.randint(-10, 10)}%")
                    else:
                        st.button(f"Acción {i+1}", key=f"card_btn_{i}")
        
        # Código de ejemplo completo
        with st.expander("Ver código de layout combinado"):
            st.code('''
# Dashboard con múltiples elementos
            
# Header con métricas
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("KPI 1", "1,234", "↑12%")

# Layout principal con sidebar
main_col, sidebar_col = st.columns([3, 1])

with sidebar_col:
    st.markdown("#### Filtros")
    period = st.radio("Período:", ["Día", "Semana", "Mes"])

with main_col:
    # Tabs con contenido
    tab1, tab2 = st.tabs(["Vista 1", "Vista 2"])
    
    with tab1:
        # Sub-columnas con gráficos
        sub_col1, sub_col2 = st.columns(2)
        
        with sub_col1:
            st.plotly_chart(fig1)
        
        with sub_col2:
            st.plotly_chart(fig2)

# Expanders para detalles adicionales
with st.expander("Más información"):
    st.write("Detalles...")
            ''', language='python')