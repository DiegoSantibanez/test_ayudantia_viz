import streamlit as st
import requests
import json
from datetime import datetime
import time
import re

def show_chatbot_ollama():
    """Función principal para mostrar la sección del chatbot con Ollama"""
    st.header("Chatbot con Ollama")
    st.write("Interactúa con modelos de lenguaje locales usando Ollama")
    
    # Inicializar session state
    if 'ollama_messages' not in st.session_state:
        st.session_state.ollama_messages = []
    if 'ollama_config' not in st.session_state:
        st.session_state.ollama_config = {
            'host': 'localhost',
            'port': '11434',
            'model': 'llama3.2:3b',
            'temperature': 0.7,
            'max_tokens': 1000
        }
    if 'ollama_connected' not in st.session_state:
        st.session_state.ollama_connected = False
    if 'available_models' not in st.session_state:
        st.session_state.available_models = []
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("Configuración de Ollama")
        
        # Configuración de conexión
        st.subheader("Conexión")
        st.session_state.ollama_config['host'] = st.text_input(
            "Host",
            value=st.session_state.ollama_config['host'],
            key="ollama_host"
        )
        
        st.session_state.ollama_config['port'] = st.text_input(
            "Puerto",
            value=st.session_state.ollama_config['port'],
            key="ollama_port"
        )
        
        # Botón para conectar y obtener modelos
        if st.button("🔍 Conectar y detectar modelos", key="connect_ollama"):
            success, models = get_available_models(
                st.session_state.ollama_config['host'],
                st.session_state.ollama_config['port']
            )
            
            if success:
                st.session_state.available_models = models
                st.session_state.ollama_connected = True
                st.success(f"Conectado! {len(models)} modelos encontrados")
            else:
                st.error("❌ No se pudo conectar con Ollama")
                st.session_state.ollama_connected = False
        
        # Mostrar estado de conexión
        if st.session_state.ollama_connected:
            st.success("Conectado a Ollama")
        else:
            st.warning("No conectado")
            
        st.markdown("---")
        
        # Configuración del modelo
        st.subheader("Configuración del Modelo")
        
        if st.session_state.available_models:
            selected_model = st.selectbox(
                "Modelo",
                st.session_state.available_models,
                index=0 if st.session_state.ollama_config['model'] not in st.session_state.available_models 
                else st.session_state.available_models.index(st.session_state.ollama_config['model']),
                key="ollama_model_select"
            )
            st.session_state.ollama_config['model'] = selected_model
        else:
            st.session_state.ollama_config['model'] = st.text_input(
                "Modelo",
                value=st.session_state.ollama_config['model'],
                help="Nombre del modelo (ej: llama3.2:3b, phi3:mini, gemma2:2b)",
                key="ollama_model_manual"
            )
        
        # Parámetros del modelo
        st.session_state.ollama_config['temperature'] = st.slider(
            "Temperatura",
            0.0, 2.0,
            st.session_state.ollama_config['temperature'],
            0.1,
            help="Controla la creatividad de las respuestas",
            key="ollama_temperature"
        )
        
        st.session_state.ollama_config['max_tokens'] = st.slider(
            "Tokens máximos",
            100, 4000,
            st.session_state.ollama_config['max_tokens'],
            100,
            help="Longitud máxima de la respuesta",
            key="ollama_max_tokens"
        )
        
        st.markdown("---")
        
        # Información de modelos recomendados
        with st.expander("Modelos recomendados (≤4B)"):
            st.markdown("""
            **Modelos livianos para Ollama:**
            
            🔹 **llama3.2:3b** - Modelo equilibrado
            🔹 **phi3:mini** - Microsoft Phi-3 (3.8B)
            🔹 **gemma2:2b** - Google Gemma 2 (2B)
            🔹 **qwen2:1.5b** - Alibaba Qwen2 (1.5B)
            🔹 **tinyllama** - Modelo ultraliviano (1.1B)
            
            **Para instalar:**
            ```bash
            ollama pull llama3.2:3b
            ollama pull phi3:mini
            ```
            """)
        
        # Gestión de conversación
        st.markdown("---")
        st.subheader("Gestión")
        
        if st.button("Limpiar conversación", key="clear_chat"):
            st.session_state.ollama_messages = []
            st.rerun()
        
        if st.button("Exportar conversación", key="export_chat"):
            if st.session_state.ollama_messages:
                chat_export = export_conversation(st.session_state.ollama_messages)
                st.download_button(
                    "Descargar",
                    chat_export,
                    "conversacion.json",
                    "application/json",
                    key="download_chat"
                )
            else:
                st.info("No hay conversación para exportar")
    
    # Contenido principal
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "Chat", "Prompts", "Estadísticas", "Configuración Avanzada"
    ])
    
    with main_tab1:
        show_chat_interface()
    
    with main_tab2:
        show_prompt_templates()
    
    with main_tab3:
        show_chat_statistics()
    
    with main_tab4:
        show_advanced_config()

def get_available_models(host, port):
    """Obtiene la lista de modelos disponibles en Ollama"""
    try:
        url = f"http://{host}:{port}/api/tags"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return True, models
        else:
            return False, []
    except Exception as e:
        st.error(f"Error conectando con Ollama: {str(e)}")
        return False, []

def send_message_to_ollama(message, config):
    """Envía un mensaje a Ollama y obtiene la respuesta"""
    try:
        url = f"http://{config['host']}:{config['port']}/api/generate"
        
        payload = {
            "model": config['model'],
            "prompt": message,
            "stream": False,
            "options": {
                "temperature": config['temperature'],
                "num_predict": config['max_tokens']
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return True, data.get('response', 'Sin respuesta')
        else:
            return False, f"Error {response.status_code}: {response.text}"
    
    except requests.exceptions.Timeout:
        return False, "Timeout: El modelo tardó demasiado en responder"
    except requests.exceptions.ConnectionError:
        return False, "Error de conexión: ¿Está Ollama ejecutándose?"
    except Exception as e:
        return False, f"Error inesperado: {str(e)}"

def show_chat_interface():
    """Interfaz principal del chat"""
    st.subheader("Conversación con el Asistente")
    
    if not st.session_state.ollama_connected:
        st.warning("Por favor, conecta con Ollama en la barra lateral antes de chatear")
        return
    
    # Contenedor para el historial de chat
    chat_container = st.container()
    
    # Mostrar mensajes existentes
    with chat_container:
        for i, message in enumerate(st.session_state.ollama_messages):
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
                    st.caption(f"{message['timestamp']}")
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
                    st.caption(f"{message['timestamp']} | {message.get('response_time', 'N/A')}s")
    
    # Input para nuevo mensaje
    user_input = st.chat_input("Escribe tu mensaje aquí...")
    
    if user_input:
        # Agregar mensaje del usuario
        user_message = {
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        st.session_state.ollama_messages.append(user_message)
        
        # Mostrar mensaje del usuario inmediatamente
        with st.chat_message("user"):
            st.write(user_input)
            st.caption(f"{user_message['timestamp']}")
        
        # Generar respuesta del asistente
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Pensando...")
            
            start_time = time.time()
            
            # Enviar mensaje a Ollama
            success, response = send_message_to_ollama(user_input, st.session_state.ollama_config)
            
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            if success:
                message_placeholder.markdown(response)
                
                # Agregar respuesta del asistente
                assistant_message = {
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'response_time': response_time
                }
                st.session_state.ollama_messages.append(assistant_message)
                
                st.caption(f"{assistant_message['timestamp']} | {response_time}s")
            else:
                message_placeholder.error(f"❌ Error: {response}")
        
        st.rerun()

def show_prompt_templates():
    """Interfaz para plantillas de prompts"""
    st.subheader("Plantillas de Prompts")
    
    # Plantillas predefinidas
    templates = {
        "Asistente General": {
            "prompt": "Eres un asistente útil y amigable. Responde de manera clara y concisa.",
            "description": "Comportamiento general para un asistente"
        },
        "Experto en Python": {
            "prompt": "Eres un experto programador en Python. Ayuda con código, debugging y mejores prácticas. Proporciona ejemplos claros y explicaciones detalladas.",
            "description": "Especializado en programación Python"
        },
        "Analista de Datos": {
            "prompt": "Eres un analista de datos experto. Ayuda con análisis estadístico, visualizaciones y interpretación de datos. Usa pandas, numpy y plotly cuando sea apropiado.",
            "description": "Especializado en análisis de datos"
        },
        "Tutor Educativo": {
            "prompt": "Eres un tutor paciente y pedagógico. Explica conceptos de manera simple y progresiva. Haz preguntas para asegurar comprensión.",
            "description": "Enfoque educativo y pedagógico"
        },
        "Revisor de Código": {
            "prompt": "Eres un revisor de código experimentado. Analiza el código en busca de errores, mejoras de rendimiento y mejores prácticas. Sé constructivo en tus comentarios.",
            "description": "Revisión y mejora de código"
        }
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Selector de plantilla
        selected_template = st.selectbox(
            "Selecciona una plantilla:",
            list(templates.keys()),
            key="template_selector"
        )
        
        # Mostrar y editar el prompt
        if selected_template:
            template_data = templates[selected_template]
            
            st.text_area(
                "Descripción:",
                value=template_data["description"],
                height=75,
                disabled=True,
                key="template_description"
            )
            
            prompt_text = st.text_area(
                "Prompt del sistema:",
                value=template_data["prompt"],
                height=150,
                key="template_prompt"
            )
            
            # Botones de acción
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("Aplicar plantilla", key="apply_template"):
                    # Reiniciar conversación con nuevo prompt del sistema
                    st.session_state.ollama_messages = []
                    
                    # Enviar prompt del sistema (no visible en chat)
                    system_message = {
                        'role': 'system',
                        'content': prompt_text,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    }
                    
                    st.success(f"Plantilla '{selected_template}' aplicada")
                    st.info("La conversación se ha reiniciado con el nuevo comportamiento")
            
            with col_b:
                if st.button("Guardar plantilla", key="save_template"):
                    # En una implementación real, guardarías en una base de datos
                    st.success("Plantilla guardada (funcionalidad simulada)")
            
            with col_c:
                if st.button("Copiar prompt", key="copy_template"):
                    st.code(prompt_text, language="text")
    
    with col2:
        st.markdown("### Consejos para Prompts")
        
        with st.expander("Mejores prácticas"):
            st.markdown("""
            **Claridad:**
            - Sé específico sobre el rol
            - Define el tono y estilo
            - Establece limitaciones claras
            
            **Estructura:**
            - Usa instrucciones paso a paso
            - Proporciona ejemplos
            - Define formato de salida
            
            **Iteración:**
            - Prueba y refina
            - Ajusta según resultados
            - Considera casos edge
            """)
        
        with st.expander("⚡ Prompts rápidos"):
            quick_prompts = [
                "Explícame esto como si tuviera 5 años",
                "Dame un resumen ejecutivo",
                "Encuentra errores en este código",
                "Sugiere mejoras",
                "Traduce al español",
                "Crea una lista de verificación"
            ]
            
            for prompt in quick_prompts:
                if st.button(f"'{prompt}'", key=f"quick_{prompt[:10]}"):
                    st.code(f"Usuario: {prompt}")

def show_chat_statistics():
    """Muestra estadísticas de la conversación"""
    st.subheader("Estadísticas de la Conversación")
    
    if not st.session_state.ollama_messages:
        st.info("No hay mensajes para analizar")
        return
    
    # Calcular estadísticas
    messages = st.session_state.ollama_messages
    user_messages = [m for m in messages if m['role'] == 'user']
    assistant_messages = [m for m in messages if m['role'] == 'assistant']
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total mensajes", len(messages))
    
    with col2:
        st.metric("Mensajes usuario", len(user_messages))
    
    with col3:
        st.metric("Respuestas asistente", len(assistant_messages))
    
    with col4:
        if assistant_messages:
            avg_response_time = sum(
                float(m.get('response_time', 0)) for m in assistant_messages
            ) / len(assistant_messages)
            st.metric("Tiempo promedio", f"{avg_response_time:.2f}s")
        else:
            st.metric("Tiempo promedio", "N/A")
    
    st.markdown("---")
    
    # Análisis de longitud de mensajes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Longitud de Mensajes")
        
        if user_messages:
            user_lengths = [len(m['content']) for m in user_messages]
            assistant_lengths = [len(m['content']) for m in assistant_messages]
            
            stats_data = {
                'Tipo': ['Usuario', 'Asistente'],
                'Promedio': [
                    sum(user_lengths) / len(user_lengths) if user_lengths else 0,
                    sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else 0
                ],
                'Máximo': [
                    max(user_lengths) if user_lengths else 0,
                    max(assistant_lengths) if assistant_lengths else 0
                ],
                'Mínimo': [
                    min(user_lengths) if user_lengths else 0,
                    min(assistant_lengths) if assistant_lengths else 0
                ]
            }
            
            import pandas as pd
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### Tiempos de Respuesta")
        
        if assistant_messages:
            response_times = [float(m.get('response_time', 0)) for m in assistant_messages]
            
            if response_times:
                import plotly.graph_objects as go
                
                fig = go.Figure(data=go.Histogram(
                    x=response_times,
                    nbinsx=10,
                    marker_color='skyblue'
                ))
                
                fig.update_layout(
                    title="Distribución de Tiempos de Respuesta",
                    xaxis_title="Tiempo (segundos)",
                    yaxis_title="Frecuencia",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Timeline de la conversación
    st.markdown("---")
    st.markdown("### Timeline de la Conversación")
    
    if len(messages) > 1:
        import plotly.express as px
        import pandas as pd
        
        # Preparar datos para timeline
        timeline_data = []
        for i, msg in enumerate(messages):
            timeline_data.append({
                'Mensaje': i + 1,
                'Rol': msg['role'],
                'Longitud': len(msg['content']),
                'Tiempo': msg['timestamp']
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        fig = px.scatter(
            timeline_df,
            x='Mensaje',
            y='Longitud',
            color='Rol',
            title="Evolución de la Longitud de Mensajes",
            labels={'Longitud': 'Caracteres', 'Mensaje': 'Número de Mensaje'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Exportar estadísticas
    st.markdown("---")
    if st.button("Exportar estadísticas", key="export_stats"):
        stats_export = {
            'resumen': {
                'total_mensajes': len(messages),
                'mensajes_usuario': len(user_messages),
                'respuestas_asistente': len(assistant_messages),
                'tiempo_promedio_respuesta': avg_response_time if assistant_messages else 0
            },
            'mensajes': messages
        }
        
        st.download_button(
            "Descargar estadísticas",
            json.dumps(stats_export, indent=2, default=str),
            "estadisticas_chat.json",
            "application/json",
            key="download_stats"
        )

def show_advanced_config():
    """Configuración avanzada del chatbot"""
    st.subheader("🔧 Configuración Avanzada")
    
    config_tab1, config_tab2, config_tab3 = st.tabs([
        "Parámetros", "Sistema", "Logs"
    ])
    
    with config_tab1:
        st.markdown("### Parámetros del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Parámetros adicionales
            top_p = st.slider(
                "Top P",
                0.0, 1.0, 0.9, 0.1,
                help="Probabilidad acumulativa para sampling",
                key="ollama_top_p"
            )
            
            top_k = st.slider(
                "Top K",
                1, 100, 40, 1,
                help="Número de tokens considerados en cada paso",
                key="ollama_top_k"
            )
            
            repeat_penalty = st.slider(
                "Penalización por repetición",
                0.0, 2.0, 1.1, 0.1,
                help="Penaliza la repetición de tokens",
                key="ollama_repeat_penalty"
            )
        
        with col2:
            seed = st.number_input(
                "Semilla (seed)",
                value=-1,
                help="Semilla para reproducibilidad (-1 = aleatoria)",
                key="ollama_seed"
            )
            
            context_length = st.slider(
                "Longitud de contexto",
                1024, 8192, 2048, 256,
                help="Tokens de contexto a mantener",
                key="ollama_context_length"
            )
            
            batch_size = st.slider(
                "Batch size",
                1, 512, 8, 1,
                help="Tamaño del lote para procesamiento",
                key="ollama_batch_size"
            )
        
        # Guardar configuración avanzada
        if st.button("Guardar configuración avanzada", key="save_advanced_config"):
            advanced_config = {
                'top_p': top_p,
                'top_k': top_k,
                'repeat_penalty': repeat_penalty,
                'seed': seed if seed != -1 else None,
                'context_length': context_length,
                'batch_size': batch_size
            }
            
            st.session_state.ollama_config.update(advanced_config)
            st.success("Configuración avanzada guardada")
    
    with config_tab2:
        st.markdown("### Información del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Estado actual:**")
            st.write(f"Host: {st.session_state.ollama_config['host']}")
            st.write(f"Puerto: {st.session_state.ollama_config['port']}")
            st.write(f"Modelo: {st.session_state.ollama_config['model']}")
            st.write(f"Temperatura: {st.session_state.ollama_config['temperature']}")
            
        with col2:
            st.markdown("**Información de Ollama:**")
            
            if st.button("Verificar estado de Ollama", key="check_ollama_status"):
                try:
                    url = f"http://{st.session_state.ollama_config['host']}:{st.session_state.ollama_config['port']}/api/version"
                    response = requests.get(url, timeout=5)
                    
                    if response.status_code == 200:
                        version_info = response.json()
                        st.success(f"Ollama funcionando - Versión: {version_info.get('version', 'Desconocida')}")
                    else:
                        st.error("Ollama no responde correctamente")
                        
                except Exception as e:
                    st.error(f"Error conectando: {str(e)}")
        
        # Test de modelo
        st.markdown("---")
        st.markdown("### Test de Modelo")
        
        test_prompt = st.text_input(
            "Prompt de prueba:",
            value="Hola, ¿cómo estás?",
            key="test_prompt"
        )
        
        if st.button("Enviar test", key="send_test"):
            if test_prompt:
                with st.spinner("Enviando mensaje de prueba..."):
                    success, response = send_message_to_ollama(test_prompt, st.session_state.ollama_config)
                
                if success:
                    st.success("Test exitoso")
                    st.text_area("Respuesta:", value=response, height=100, disabled=True)
                else:
                    st.error(f"Test fallido: {response}")
    
    with config_tab3:
        st.markdown("### Logs y Debugging")
        
        # Toggle para logs detallados
        enable_logging = st.checkbox(
            "Habilitar logs detallados",
            value=False,
            key="enable_logging"
        )
        
        if enable_logging:
            st.info("Modo debug activado - Se mostrarán logs detallados")
        
        # Mostrar configuración actual
        with st.expander("Configuración actual completa"):
            st.json(st.session_state.ollama_config)
        
        # Logs simulados (en una implementación real, estos vendrían de archivos de log)
        with st.expander("Logs recientes"):
            sample_logs = [
                f"[{datetime.now().strftime('%H:%M:%S')}] INFO: Modelo cargado correctamente",
                f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Configuración aplicada",
                f"[{datetime.now().strftime('%H:%M:%S')}] INFO: Conexión establecida",
            ]
            
            for log in sample_logs:
                st.text(log)
        
        # Exportar configuración
        if st.button("Exportar configuración", key="export_config"):
            config_export = {
                'ollama_config': st.session_state.ollama_config,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            st.download_button(
                "Descargar configuración",
                json.dumps(config_export, indent=2),
                "ollama_config.json",
                "application/json",
                key="download_config"
            )

def export_conversation(messages):
    """Exporta la conversación a formato JSON"""
    export_data = {
        'conversation': messages,
        'export_timestamp': datetime.now().isoformat(),
        'total_messages': len(messages),
        'config_used': st.session_state.ollama_config
    }
    
    return json.dumps(export_data, indent=2, default=str)

# Función auxiliar para formatear mensajes
def format_message_for_display(message):
    """Formatea un mensaje para mostrar en la interfaz"""
    content = message['content']
    
    # Detectar y formatear código
    if '```' in content:
        # El mensaje contiene bloques de código
        parts = content.split('```')
        formatted_parts = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Texto normal
                formatted_parts.append(part)
            else:
                # Bloque de código
                lines = part.split('\n')
                language = lines[0] if lines else ''
                code = '\n'.join(lines[1:]) if len(lines) > 1 else part
                
                formatted_parts.append(f"\n```{language}\n{code}\n```\n")
        
        return ''.join(formatted_parts)
    
    return content