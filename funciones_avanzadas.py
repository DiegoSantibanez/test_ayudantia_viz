import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import json
import streamlit.components.v1 as components



def show_funciones_avanzadas():
    """Sección de funciones avanzadas de Streamlit"""
    st.header("Funciones Avanzadas")
    st.write("Características avanzadas y técnicas de Streamlit para aplicaciones profesionales")
    
    # Tabs para organizar las funciones avanzadas
    tabs = st.tabs([
        "Progress & Status", 
        "Streaming", 
        "Custom Components",
        "Notificaciones",
        "Callbacks & Eventos",
        # "Conexiones API",
        "Otras cosas"
    ])
    
    with tabs[0]:
        show_progress_status()
    
    with tabs[1]:
        show_streaming()
    
    with tabs[2]:
        show_custom_components()
    
    with tabs[3]:
        show_notifications()
    
    with tabs[4]:
        show_callbacks_events()
    
    # with tabs[5]:
    #     show_api_connections()
    with tabs[5]:
        show_html()



def show_progress_status():
    """Demostración de Progress bars y Status"""
    st.subheader("Progress Bars y Status")
    st.write("Comunica el progreso y estado de las operaciones")
    
    # Ejemplo 1: Progress bar básico
    st.markdown("### Progress Bar Básico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("Duración (segundos):", 1, 10, 3, key="progress_duration")
        
        if st.button("Iniciar Progreso"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(duration * 10):
                progress = (i + 1) / (duration * 10)
                progress_bar.progress(progress)
                status_text.text(f'Progreso: {progress:.0%}')
                time.sleep(0.1)
            
            progress_bar.empty()
            status_text.empty()
            st.success("¡Completado!")
    
    with col2:
        st.markdown("#### Progress con Spinner")
        
        if st.button("Proceso con Spinner"):
            with st.spinner('Procesando datos...'):
                time.sleep(3)
            st.success("¡Proceso completado!")
    
    with st.expander("Ver código de progress básico"):
        st.code('''
# Progress bar
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

# Spinner
with st.spinner('Cargando...'):
    time.sleep(2)
st.success("¡Listo!")
        ''', language='python')
    
    st.markdown("---")
    
    # Ejemplo 2: Progress multi-etapa
    st.markdown("### Progress Multi-etapa")
    
    if st.button("Proceso Multi-etapa"):
        stages = [
            ("Iniciando proceso", 0.1),
            ("Cargando datos", 0.3),
            ("Procesando información", 0.6),
            ("Generando reportes", 0.8),
            ("Finalizando", 1.0)
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for stage_name, progress_value in stages:
            status_text.text(f"{stage_name}...")
            progress_bar.progress(progress_value)
            time.sleep(1)
        
        progress_bar.empty()
        status_text.empty()
        st.success("Todos los procesos completados")
        st.balloons()
    
    st.markdown("---")
    
    # Ejemplo 3: Progress con métricas en tiempo real
    st.markdown("### Progress con Métricas en Tiempo Real")
    
    if st.button("Simulación de Entrenamiento ML"):
        epochs = 20
        
        # Containers para las métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy_metric = st.empty()
        with col2:
            loss_metric = st.empty()
        with col3:
            epoch_metric = st.empty()
        
        # Progress bar y status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Chart para mostrar evolución
        chart_data = pd.DataFrame(columns=['epoch', 'accuracy', 'loss'])
        chart_placeholder = st.empty()
        
        # Simular entrenamiento
        for epoch in range(epochs):
            # Simular mejora gradual
            accuracy = 0.5 + (epoch / epochs) * 0.4 + random.uniform(-0.02, 0.02)
            loss = 2.0 - (epoch / epochs) * 1.5 + random.uniform(-0.1, 0.1)
            
            # Actualizar métricas
            accuracy_metric.metric("Precisión", f"{accuracy:.3f}", f"{accuracy-0.5:.3f}")
            loss_metric.metric("Pérdida", f"{loss:.3f}", f"{-(loss-2.0):.3f}")
            epoch_metric.metric("Época", f"{epoch+1}/{epochs}")
            
            # Actualizar progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Entrenando... Época {epoch+1}/{epochs}")
            
            # Actualizar chart
            new_row = pd.DataFrame({
                'epoch': [epoch+1],
                'accuracy': [accuracy],
                'loss': [loss]
            })
            chart_data = pd.concat([chart_data, new_row], ignore_index=True)
            
            # Crear gráfico actualizado
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart_data['epoch'], y=chart_data['accuracy'], 
                                   name='Precisión', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=chart_data['epoch'], y=chart_data['loss'], 
                                   name='Pérdida', line=dict(color='red'), yaxis='y2'))
            
            fig.update_layout(
                title="Evolución del Entrenamiento",
                xaxis_title="Época",
                yaxis=dict(title="Precisión", side="left"),
                yaxis2=dict(title="Pérdida", side="right", overlaying="y"),
                height=300
            )
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            time.sleep(0.3)
        
        # Limpiar elementos temporales
        progress_bar.empty()
        status_text.empty()
        st.success("¡Entrenamiento completado!")
    
    st.markdown("---")
    
    # Ejemplo 4: Status alerts y notificaciones
    st.markdown("### Status Alerts y Notificaciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tipos de Alertas")
        
        if st.button("✅ Success"):
            st.success("Operación completada exitosamente")
        
        if st.button("ℹ️ Info"):
            st.info("Esta es información importante")
        
        if st.button("⚠️ Warning"):
            st.warning("Ten cuidado con esta acción")
        
        if st.button("❌ Error"):
            st.error("Ha ocurrido un error")
    
    with col2:
        st.markdown("#### Toast Notifications")
        
        if st.button("🎈 Balloons"):
            st.balloons()
        
        if st.button("❄️ Snow"):
            st.snow()
        
        if st.button("🎊 Toast Success"):
            st.toast("¡Operación exitosa!", icon="✅")
        
        if st.button("⚠️ Toast Warning"):
            st.toast("Atención requerida", icon="⚠️")


def show_streaming():
    """Demostración de Streaming y contenido dinámico"""
    st.subheader("Streaming y Contenido Dinámico")
    st.write("Actualiza contenido en tiempo real")
    
    # Ejemplo 1: Reloj en tiempo real
    st.markdown("### Reloj en Tiempo Real")
    
    clock_placeholder = st.empty()
    start_clock = st.checkbox("Iniciar reloj")
    
    if start_clock:
        if 'clock_running' not in st.session_state:
            st.session_state.clock_running = True
        
        while st.session_state.clock_running and start_clock:
            current_time = datetime.now()
            clock_placeholder.markdown(f"""
            <div style="font-size: 2em; text-align: center; color: #1f77b4;">
                {current_time.strftime('%Y-%m-%d')}<br>
                {current_time.strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
    else:
        st.session_state.clock_running = False
        clock_placeholder.empty()
    
    st.markdown("---")
    
    # Ejemplo 2: Datos en streaming
    st.markdown("### Datos en Streaming")
    
    if st.button("Iniciar Stream de Datos"):
        # Containers para los datos
        chart_container = st.empty()
        metrics_container = st.empty()
        data_container = st.empty()
        
        # Inicializar datos
        stream_data = pd.DataFrame(columns=['timestamp', 'value'])
        
        for i in range(30):  # 30 actualizaciones
            # Generar nuevo punto de datos
            new_value = 50 + 20 * np.sin(i * 0.3) + random.uniform(-5, 5)
            new_timestamp = datetime.now() - timedelta(seconds=30-i)
            
            # Agregar a dataset
            new_row = pd.DataFrame({
                'timestamp': [new_timestamp],
                'value': [new_value]
            })
            stream_data = pd.concat([stream_data, new_row], ignore_index=True)
            
            # Mantener solo últimos 20 puntos
            if len(stream_data) > 20:
                stream_data = stream_data.tail(20).reset_index(drop=True)
            
            # Actualizar gráfico
            fig = px.line(stream_data, x='timestamp', y='value', 
                         title="Datos en Tiempo Real")
            fig.update_layout(height=300)
            chart_container.plotly_chart(fig, use_container_width=True)
            
            # Actualizar métricas
            if len(stream_data) > 0:
                current_value = stream_data['value'].iloc[-1]
                avg_value = stream_data['value'].mean()
                
                with metrics_container.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Valor Actual", f"{current_value:.2f}")
                    with col2:
                        st.metric("Promedio", f"{avg_value:.2f}")
                    with col3:
                        st.metric("Puntos", len(stream_data))
            
            # Mostrar datos más recientes
            with data_container.container():
                st.write("**Últimos 5 valores:**")
                recent_data = stream_data.tail(5)[['timestamp', 'value']].copy()
                recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%H:%M:%S')
                st.dataframe(recent_data, use_container_width=True, hide_index=True)
            
            time.sleep(0.5)
        
        st.success("🎉 Stream completado")



def show_custom_components():
    """Demostración de Custom Components"""
    st.subheader("Custom Components")
    st.write("Componentes personalizados con HTML/CSS/JavaScript")
    
    # Ejemplo 1: Componente HTML personalizado
    st.markdown("###  HTML/CSS Personalizado")
    
    # CSS personalizado
    st.markdown("""
    <style>
    .custom-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .pulse-button {
        background: #ff6b6b;
        border: none;
        border-radius: 50px;
        color: white;
        padding: 15px 30px;
        font-size: 16px;
        cursor: pointer;
        animation: pulse 2s infinite;
        margin: 10px;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .progress-ring {
        width: 120px;
        height: 120px;
        border: 8px solid #f3f3f3;
        border-top: 8px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Card personalizada
    st.markdown("""
    <div class="custom-card">
        <h2>Componente Personalizado</h2>
        <p>Este es un ejemplo de diseño personalizado con CSS</p>
        <p>Gradientes, sombras y animaciones</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("---")
    
    # Ejemplo 3: Dashboard card con métricas avanzadas
    st.markdown("### Dashboard Cards Avanzado (Hechas con HTML)")
    
    # Generar datos de ejemplo
    metrics_data = {
        'sales': {'value': 125000, 'change': 12.5, 'target': 120000},
        'users': {'value': 2847, 'change': -3.2, 'target': 3000},
        'revenue': {'value': 87.5, 'change': 8.7, 'target': 85},
        'satisfaction': {'value': 4.6, 'change': 0.3, 'target': 4.5}
    }
    
    cols = st.columns(4)
    
    for i, (key, data) in enumerate(metrics_data.items()):
        with cols[i]:
            # Determinar color según el rendimiento
            if data['value'] >= data['target']:
                color = "#28a745"  # Verde
                icon = "✅"
            else:
                color = "#ffc107"  # Amarillo
                icon = "⚠️"
            
            # Card personalizada con métricas
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}20, {color}10);
                border-left: 4px solid {color};
                padding: 15px;
                border-radius: 10px;
                margin: 5px 0;
            ">
                <h4 style="margin: 0; color: {color};">{icon} {key.title()}</h4>
                <h2 style="margin: 5px 0; color: #333;">{data['value']:,}</h2>
                <p style="margin: 0; color: #666;">
                    Cambio: {data['change']:+.1f}% | Meta: {data['target']:,}
                </p>
                <div style="
                    background: #e9ecef;
                    border-radius: 10px;
                    height: 6px;
                    margin-top: 10px;
                ">
                    <div style="
                        background: {color};
                        height: 6px;
                        border-radius: 10px;
                        width: {min(100, (data['value']/data['target'])*100):.1f}%;
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    


def show_notifications():
    """Demostración de sistema de notificaciones"""
    st.subheader("Sistema de Notificaciones")
    st.write("Gestiona notificaciones y alertas de manera efectiva")
    
    # Inicializar sistema de notificaciones
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    # Ejemplo 1: Centro de notificaciones
    st.markdown("### Centro de Notificaciones")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("#### Controles")
        
        # Botones para generar notificaciones
        if st.button("Nueva Notificación"):
            notification = {
                'id': len(st.session_state.notifications),
                'type': random.choice(['info', 'success', 'warning', 'error']),
                'title': random.choice([
                    "Nuevo mensaje",
                    "Tarea completada", 
                    "Recordatorio",
                    "Alerta del sistema"
                ]),
                'message': random.choice([
                    "Tienes un nuevo mensaje en tu bandeja",
                    "La tarea se completó exitosamente",
                    "No olvides revisar tu agenda",
                    "Se detectó un problema en el sistema"
                ]),
                'timestamp': datetime.now(),
                'read': False
            }
            st.session_state.notifications.append(notification)
            st.rerun()
        
        if st.button("Limpiar Todo"):
            st.session_state.notifications = []
            st.rerun()
        
        if st.button("Marcar Todo Leído"):
            for notif in st.session_state.notifications:
                notif['read'] = True
            st.rerun()
    
    with col1:
        st.markdown("#### Notificaciones Recientes")
        
        if st.session_state.notifications:
            # Ordenar por timestamp descendente
            sorted_notifications = sorted(
                st.session_state.notifications,
                key=lambda x: x['timestamp'],
                reverse=True
            )
            
            for notif in sorted_notifications[:10]:  # Mostrar últimas 10
                # Definir colores según tipo
                colors = {
                    'info': '#17a2b8',
                    'success': '#28a745',
                    'warning': '#ffc107',
                    'error': '#dc3545'
                }
                
                icons = {
                    'info': 'ℹ️',
                    'success': '✅',
                    'warning': '⚠️',
                    'error': '❌'
                }
                
                color = colors.get(notif['type'], '#6c757d')
                icon = icons.get(notif['type'], '📄')
                opacity = '0.6' if notif['read'] else '1.0'
                
                # Card de notificación
                st.markdown(f"""
                <div style="
                    border-left: 4px solid {color};
                    background: rgba(255,255,255,0.8);
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    opacity: {opacity};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h5 style="margin: 0; color: {color};">
                            {icon} {notif['title']}
                        </h5>
                        <small style="color: #6c757d;">
                            {notif['timestamp'].strftime('%H:%M')}
                        </small>
                    </div>
                    <p style="margin: 5px 0 0 0; color: #333;">
                        {notif['message']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Botón para marcar como leído
                if not notif['read']:
                    if st.button(f"Marcar leído", key=f"read_{notif['id']}"):
                        notif['read'] = True
                        st.rerun()
        else:
            st.info("No hay notificaciones")
    
    st.markdown("---")
    
    # Ejemplo 2: Toast notifications
    st.markdown("### Toast Notifications")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎉 Success Toast"):
            st.toast("¡Operación exitosa!", icon="✅")
    
    with col2:
        if st.button("ℹ️ Info Toast"):
            st.toast("Información importante", icon="ℹ️")
    
    with col3:
        if st.button("⚠️ Warning Toast"):
            st.toast("Advertencia del sistema", icon="⚠️")
    
    with col4:
        if st.button("❌ Error Toast"):
            st.toast("Error encontrado", icon="❌")
    
    st.markdown("---")
    
    # Ejemplo 3: Alertas contextuales
    st.markdown("### Alertas Contextuales")
    
    alert_type = st.selectbox(
        "Tipo de alerta:",
        ["success", "info", "warning", "error"]
    )
    
    alert_message = st.text_input(
        "Mensaje de la alerta:",
        value="Este es un mensaje de prueba"
    )
    
    if st.button("Mostrar Alerta"):
        if alert_type == "success":
            st.success(alert_message)
        elif alert_type == "info":
            st.info(alert_message)
        elif alert_type == "warning":
            st.warning(alert_message)
        elif alert_type == "error":
            st.error(alert_message)


def show_callbacks_events():
    """Demostración de callbacks y manejo de eventos"""
    st.subheader("Callbacks y Manejo de Eventos")
    st.write("Responde a eventos y crea interacciones complejas")
    
    # Ejemplo 1: Callback con session state
    st.markdown("### Callbacks con Session State")
    
    # Inicializar contadores
    if 'click_count' not in st.session_state:
        st.session_state.click_count = 0
    
    if 'last_clicked' not in st.session_state:
        st.session_state.last_clicked = None
    
    def handle_click(button_name):
        """Callback function para manejar clicks"""
        st.session_state.click_count += 1
        st.session_state.last_clicked = button_name
        st.session_state.last_click_time = datetime.now()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Botón Rojo", key="red_btn"):
            handle_click("Rojo")
    
    with col2:
        if st.button("Botón Azul", key="blue_btn"):
            handle_click("Azul")
    
    with col3:
        if st.button("Botón Verde", key="green_btn"):
            handle_click("Verde")
    
    # Mostrar información del evento
    if st.session_state.last_clicked:
        st.success(f"Último click: Botón {st.session_state.last_clicked}")
        st.info(f"Total de clicks: {st.session_state.click_count}")
        if hasattr(st.session_state, 'last_click_time'):
            st.caption(f"Hora: {st.session_state.last_click_time.strftime('%H:%M:%S')}")
    
    # Reset button
    if st.button("Reset Contadores"):
        st.session_state.click_count = 0
        st.session_state.last_clicked = None
        st.rerun()
    
    st.markdown("---")
    
    # Ejemplo 2: Eventos de formulario
    st.markdown("### Eventos de Formulario")
    
    with st.form("event_form"):
        st.write("Formulario con validación en tiempo real")
        
        name = st.text_input("Nombre:", help="Mínimo 3 caracteres")
        email = st.text_input("Email:", help="Debe contener @")
        age = st.number_input("Edad:", min_value=0, max_value=120, value=25)
        
        # Checkboxes para términos
        terms = st.checkbox("Acepto términos y condiciones")
        newsletter = st.checkbox("Suscribirse al newsletter")
        
        submitted = st.form_submit_button("Enviar Formulario")
        
        if submitted:
            # Validación personalizada
            errors = []
            
            if len(name) < 3:
                errors.append("El nombre debe tener al menos 3 caracteres")
            
            if "@" not in email:
                errors.append("Email debe contener @")
            
            if age < 18:
                errors.append("Debe ser mayor de 18 años")
            
            if not terms:
                errors.append("Debe aceptar términos y condiciones")
            
            if errors:
                for error in errors:
                    st.error(f"{error}")
            else:
                st.success("Formulario enviado correctamente!")
                
                # Simular proceso
                with st.spinner("Procesando datos..."):
                    time.sleep(1)
                
                st.balloons()
                
                # Mostrar resumen
                st.json({
                    "nombre": name,
                    "email": email,
                    "edad": age,
                    "newsletter": newsletter,
                    "timestamp": datetime.now().isoformat()
                })
    
    st.markdown("---")
    
    # Ejemplo 3: Eventos de carga de archivos
    st.markdown("### Eventos de Carga de Archivos")
    
    uploaded_file = st.file_uploader(
        "Sube un archivo:",
        type=['txt', 'csv', 'json', 'xlsx'],
        help="Archivos soportados: TXT, CSV, JSON, XLSX"
    )
    
    if uploaded_file is not None:
        # Información del archivo
        file_details = {
            "Nombre": uploaded_file.name,
            "Tipo": uploaded_file.type,
            "Tamaño": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        st.success("Archivo cargado exitosamente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json(file_details)
        
        with col2:
            # Procesar según el tipo de archivo
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("**Vista previa del CSV:**")
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error al leer CSV: {str(e)}")
            
            elif uploaded_file.name.endswith('.txt'):
                try:
                    content = str(uploaded_file.read(), "utf-8")
                    st.write("**Contenido del archivo:**")
                    st.text_area("", content[:500] + "..." if len(content) > 500 else content, height=150)
                except Exception as e:
                    st.error(f"Error al leer archivo: {str(e)}")
            
            elif uploaded_file.name.endswith('.json'):
                try:
                    json_data = json.load(uploaded_file)
                    st.write("**Contenido JSON:**")
                    st.json(json_data)
                except Exception as e:
                    st.error(f"Error al leer JSON: {str(e)}")
    
    st.markdown("---")
    
    # Ejemplo 4: Eventos de selección
    st.markdown("### Eventos de Selección Múltiple")
    
    # Datos de ejemplo
    options = {
        "Frutas": ["Manzana", "Plátano", "Naranja", "Uvas"],
        "Colores": ["Rojo", "Azul", "Verde", "Amarillo"],
        "Animales": ["Perro", "Gato", "Ratón", "Conejo"]
    }
    
    selected_category = st.selectbox("Selecciona una categoría:", list(options.keys()))
    
    if selected_category:
        selected_items = st.multiselect(
            f"Selecciona {selected_category.lower()}:",
            options[selected_category],
            help=f"Puedes seleccionar múltiples {selected_category.lower()}"
        )
        
        if selected_items:
            st.write(f"**Has seleccionado {len(selected_items)} {selected_category.lower()}:**")
            
            # Mostrar selección en columnas
            cols = st.columns(min(len(selected_items), 4))
            
            for i, item in enumerate(selected_items):
                with cols[i % len(cols)]:
                    st.info(item)
            
            # Botón de acción
            if st.button(f"Procesar {selected_category}"):
                st.success(f"Procesando: {', '.join(selected_items)}")
                
                # Simular procesamiento
                progress_bar = st.progress(0)
                for i in range(len(selected_items)):
                    time.sleep(0.5)
                    progress_bar.progress((i + 1) / len(selected_items))
                
                st.balloons()
                st.write("¡Procesamiento completado!")


def show_html():
    # Cargar el archivo HTML
    with open('data/juego.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    st.title("Juego en HTML con JavaScript")
    components.html(html_content, height=600)
    # st.markdown(html_content, unsafe_allow_html=True)
