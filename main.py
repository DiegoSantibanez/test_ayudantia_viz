import streamlit as st

from visualizaciones import show_visualizaciones
from manejo_datos import show_manejo_datos
from layout_diseno import show_layout_diseno
from funciones_avanzadas import show_funciones_avanzadas
from exploracion_postgres import show_exploracion_postgres
from subida_datos_postgres import show_subida_postgres
from chat_ollama import show_chatbot_ollama

# Configuración de la página
st.set_page_config(
    page_title="Ayudantía Streamilt",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado (opcional)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Título principal
    st.markdown('<h1 class="main-header">Ayundatía: Streamlit</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar para navegación
    with st.sidebar:
        st.title("Navegación")
        demo_section = st.selectbox(
            "Selecciona una sección:",
            [
                "Inicio",
                "Widgets Básicos", 
                "Visualizaciones",
                "Manejo de Datos",
                "Layout y Diseño",
                "Funciones Avanzadas",
                "🐘 Explorador PostgreSQL",
                "Subida a PostgreSQL",
                "Chatbot Ollama"
            ]
        )
        
        st.markdown("---")
        
        # Información específica para el chatbot
        if demo_section == "Chatbot Ollama":
            st.markdown("### Requisitos Chatbot")
            st.warning("""
            **Necesitas tener Ollama instalado:**
            
            1. Instala Ollama desde [ollama.ai](https://ollama.ai)
            2. Ejecuta: `ollama pull llama3.2:3b`
            3. Inicia Ollama: `ollama serve`
            
            **Modelos recomendados (≤4B):**
            - llama3.2:3b
            - phi3:mini
            - gemma2:2b
            """)
    
    # Routing de secciones
    if demo_section == "Inicio":
        show_inicio()
    elif demo_section == "Widgets Básicos":
        show_widgets_basicos()
    elif demo_section == "Visualizaciones":
        show_visualizaciones()
    elif demo_section == "Manejo de Datos":
        show_manejo_datos()
    elif demo_section == "Layout y Diseño":
        show_layout_diseno()
    elif demo_section == "Funciones Avanzadas":
        show_funciones_avanzadas()
    elif demo_section == "🐘 Explorador PostgreSQL": 
        show_exploracion_postgres() 
    elif demo_section == "Subida a PostgreSQL": 
        show_subida_postgres()
    elif demo_section == "Chatbot Ollama":
        show_chatbot_ollama()

def show_inicio():
    st.subheader("¿Qué es Streamlit?")
    st.write("""
    **Streamlit** es un framework de código abierto escrito en Python que revoluciona 
    la creación de aplicaciones web para ciencia de datos y machine learning. 
    Permite a los científicos de datos y desarrolladores crear aplicaciones web 
    interactivas  y funcionales de manera rápida usando únicamente Python, sin necesidad 
    de conocimientos en HTML, CSS o JavaScript.
    
    Su filosofía principal es **"Scripts to Apps"** - convertir scripts de Python 
    en aplicaciones web compartibles con solo unas pocas líneas de código.
    """)
    
    st.subheader("Características principales:")
    st.markdown("""
    - **Simplicidad**: Código Python puro, sin frontend complejo
    - **Rapidez**: Prototipado rápido y desarrollo ágil
    - **Interactividad**: Widgets reactivos y actualizaciones en tiempo real
    - **Visualizaciones**: Integración nativa con Matplotlib y Plotly
    - **Hot reload**: Actualización automática al guardar cambios
    - **Despliegue fácil**: Desde local hasta la nube en minutos
    """)
    
    
    # Historia de Streamlit
    st.markdown("---")
    st.subheader("Historia de Streamlit")
    
    # Timeline con expanders
    with st.expander("**2018-2019: Los Inicios**", expanded=True):
        st.write("""
        **Fundación**: Streamlit fue fundado en 2018 por **Adrien Treuille**, **Thiago Teixeira** 
        y **Amanda Kelly** en San Francisco, California.
        
        **Problema identificado**: Los fundadores, con experiencia en Google X y otras empresas tech, 
        notaron que los científicos de datos tenían dificultades para compartir su trabajo de manera 
        interactiva sin depender de equipos de desarrollo web.
        
        **Visión**: Crear una herramienta que permitiera a cualquier persona con conocimientos de 
        Python crear aplicaciones web hermosas y funcionales sin escribir una sola línea de HTML o JavaScript.
        """)
    
    with st.expander("**2019: Lanzamiento Público**"):
        st.write("""
        **Octubre 2019**: Lanzamiento público de Streamlit como proyecto de código abierto.
        
        **Recepción**: La comunidad de desarrolladores y científicos de datos recibió la herramienta 
        con gran entusiasmo. En pocas semanas, el repositorio de GitHub alcanzó miles de estrellas. 
                 Lo han adoptado empresas como Uber, Spotify, NASA y la World Health Organization
        """)
    
    with st.expander("**2020: Inversión y Crecimiento**"):
        st.write("""
        **Enero 2020**: Streamlit recaudó **$21 millones** en financiación Serie A, liderada por GGV Capital.
        
        **Crecimiento exponencial**: 
        - Miles de aplicaciones creadas por la comunidad
        
        **Streamlit Sharing**: Lanzamiento de la plataforma gratuita para hospedar aplicaciones Streamlit.
        """)
    
    with st.expander("**2021: Adquisición por Snowflake**"):
        st.write("""
        **Marzo 2021**: **Snowflake adquirió Streamlit por $800 millones**, una de las adquisiciones 
        más grandes en el espacio de herramientas para desarrolladores.
        
        **Integración estratégica**: La adquisición permitió una integración más profunda con 
        plataformas de datos empresariales y aceleró el desarrollo de nuevas características.
        
        **Streamlit Cloud**: Evolución de Streamlit Sharing con más funcionalidades empresariales.
        """)
    
    with st.expander("**2022-2024: Madurez y Expansión**"):
        st.write("""
        **Nuevas características**:
        - Streamlit Components: Componentes personalizados con HTML/JS
        - Session State: Gestión avanzada de estado
        - Multipage Apps: Aplicaciones multi-página nativas
        - Streamlit Elements: Integraciones avanzadas
        
        **Ecosistema**: Crecimiento de un ecosistema robusto con miles de componentes de terceros 
        y integraciones con las principales librerías de Python.
        
        """)
    
    # Sección de Demo
    st.markdown("---")

    st.subheader("Impacto en la Industria")
    st.markdown("""
    **Democratización**: Streamlit ha democratizado la creación de aplicaciones web para 
    científicos de datos, permitiendo que se enfoquen en la lógica de negocio en lugar 
    de la infraestructura web.
    
    **Casos de uso populares**:
    - Dashboards de análisis de datos
    - Demos de modelos de ML
    - Herramientas de visualización
    - Aplicaciones de exploración de datos
    - Chatbots y asistentes IA
    """)

def show_widgets_basicos():
    """Sección de widgets básicos"""
    st.header("Widgets Básicos")
    st.write("Aquí puedes experimentar con los diferentes widgets de Streamlit")
    
    st.subheader("Button elements")
    
    # Crear un layout de 2 columnas para organizar los botones
    col1, col2 = st.columns(2)
    
    with col1:
        # Button básico
        st.markdown("### Button")
        st.write("Display a button widget.")
        
        if st.button("Click me"):
            st.success("¡Botón clickeado!")
        
        # Mostrar el código
        with st.expander("Ver código"):
            st.code('''clicked = st.button("Click me")''', language='python')
        
        st.markdown("---")
        
        # Link button
        st.markdown("### Link button")
        st.write("Display a link button.")
        
        st.link_button("Go to gallery", "https://streamlit.io/gallery")
        
        # Mostrar el código
        with st.expander("Ver código"):
            st.code('''st.link_button("Go to gallery", "https://streamlit.io/gallery")''', language='python')
    
    with col2:
        # Download button
        st.markdown("### Download button")
        st.write("Display a download button widget.")
        
        # Crear datos de ejemplo para descargar
        sample_data = "Nombre,Edad,Ciudad\nJuan,25,Madrid\nAna,30,Barcelona\nLuis,22,Sevilla"
        
        st.download_button(
            label="Download file",
            data=sample_data,
            file_name="datos_ejemplo.csv",
            mime="text/csv"
        )
        
        # Mostrar el código
        with st.expander("Ver código"):
            st.code('''st.download_button("Download file", data=csv_data, file_name="data.csv")''', language='python')
        
        st.markdown("---")
        
        # Page link
        st.markdown("### Page link")
        st.write("Display a link to another page in a multipage app.")
        
        # Como esto es una demo, simulamos el page_link
        if st.button("Home"):
            st.info("En una app multipágina, esto te llevaría a la página Home")
        if st.button("1️Page 1"):
            st.info("En una app multipágina, esto te llevaría a Page 1")
        
        # Mostrar el código
        with st.expander("Ver código"):
            st.code('''st.page_link("app.py", label="Home")
st.page_link("pages/profile.py", label="Page 1")''', language='python')
    
    # Form button en una sección separada
    st.markdown("---")
    st.markdown("### Form button")
    st.write("Display a form submit button. For use with st.form.")
    
    with st.form("my_form"):
        name = st.text_input("Nombre")
        age = st.number_input("Edad", min_value=0, max_value=120, value=25)
        
        # El form_submit_button solo funciona dentro de un form
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            st.success(f"Formulario enviado! Nombre: {name}, Edad: {age}")
    
    # Mostrar el código del form
    with st.expander("Ver código del formulario"):
        st.code('''with st.form("my_form"):
    name = st.text_input("Nombre")
    age = st.number_input("Edad")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.success(f"Datos: {name}, {age}")''', language='python')
    
    # Sección adicional con otros widgets básicos
    st.markdown("---")
    st.subheader("Otros Widgets Básicos")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Text input
        st.markdown("### Text Input")
        text_value = st.text_input("Escribe algo:", placeholder="Tu texto aquí...")
        if text_value:
            st.write(f"Escribiste: {text_value}")
        
        # Number input
        st.markdown("### Number Input")
        number_value = st.number_input("Selecciona un número:", min_value=0, max_value=100, value=50)
        st.write(f"Número seleccionado: {number_value}")
        
        # Slider
        st.markdown("### Slider")
        slider_value = st.slider("Selecciona un valor:", 0, 100, 25)
        st.write(f"Valor del slider: {slider_value}")
    
    with col4:
        # Selectbox
        st.markdown("### Select Box")
        option = st.selectbox("Elige una opción:", ["Opción 1", "Opción 2", "Opción 3"])
        st.write(f"Seleccionaste: {option}")
        
        # Checkbox
        st.markdown("### Checkbox")
        checkbox_value = st.checkbox("Marcar esta casilla")
        if checkbox_value:
            st.write("Casilla marcada")
        
        # Radio buttons
        st.markdown("### Radio Buttons")
        radio_value = st.radio("Elige una opción:", ["A", "B", "C"])
        st.write(f"Opción seleccionada: {radio_value}")
    
    # Multiselect
    st.markdown("### Multi Select")
    multiselect_values = st.multiselect(
        "Selecciona múltiples opciones:",
        ["Python", "JavaScript", "Java", "C++", "Go", "Rust"],
        default=["Python"]
    )
    if multiselect_values:
        st.write(f"Lenguajes seleccionados: {', '.join(multiselect_values)}")
    
    # Date and Time inputs
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("### Date Input")
        date_value = st.date_input("Selecciona una fecha:")
        st.write(f"Fecha seleccionada: {date_value}")
    
    with col6:
        st.markdown("### Time Input")
        time_value = st.time_input("Selecciona una hora:")
        st.write(f"Hora seleccionada: {time_value}")
    
    # Color picker
    st.markdown("### Color Picker")
    color_value = st.color_picker("Elige un color:", "#00f900")
    st.write(f"Color seleccionado: {color_value}")
    
    # File uploader
    st.markdown("### File Uploader")
    uploaded_file = st.file_uploader("Sube un archivo:", type=['txt', 'csv', 'xlsx'])
    if uploaded_file is not None:
        st.success(f"Archivo subido: {uploaded_file.name}")
        st.write(f"Tamaño: {uploaded_file.size} bytes")



# Ejecutar la aplicación
if __name__ == "__main__":
    main()