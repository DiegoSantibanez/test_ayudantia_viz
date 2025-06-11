import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import re
import json



def parse_rgb_color(color_str):
    """
    Convierte un string de color RGB a tupla de enteros.
    
    Args:
        color_str: String en formato 'rgb(r, g, b)'
    
    Returns:
        Tupla con valores RGB (r, g, b)
    
    Raises:
        ValueError: Si el formato del color no es válido
    """
    # Usar regex para extraer los números del formato rgb(r, g, b)
    match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str.strip())
    if not match:
        raise ValueError(f"Formato de color inválido: {color_str}")
    
    r, g, b = map(int, match.groups())
    
    # Validar que los valores estén en el rango correcto
    for val in [r, g, b]:
        if not 0 <= val <= 255:
            raise ValueError(f"Valores RGB deben estar entre 0 y 255, encontrado: {val}")
    
    return (r, g, b)

def rgb_to_hex(r, g, b):
    """Convierte valores RGB a formato hexadecimal."""
    return f"#{r:02x}{g:02x}{b:02x}"

def interpolar_color(pos, colores):
    """
    Interpola un color en una posición específica de una escala de colores.
    
    Args:
        pos: Posición para interpolar (0.0 a 1.0)
        colores: Escala de colores en formato [[posición, 'rgb(r,g,b)'], ...]
                Ejemplo: [[0, 'rgb(240, 242, 246)'], [1, 'rgb(26, 32, 44)']]
    
    Returns:
        String con el color interpolado en formato 'rgb(r, g, b)'
    
    Raises:
        ValueError: Si los parámetros no son válidos
    """
    if not colores:
        raise ValueError("La lista de colores no puede estar vacía")
    
    if not 0 <= pos <= 1:
        raise ValueError(f"La posición debe estar entre 0 y 1, recibido: {pos}")
    
    # Ordenar colores por posición para asegurar orden correcto
    colores_ordenados = sorted(colores, key=lambda x: x[0])
    
    # Si la posición está antes del primer color, retornar el primero
    if pos <= colores_ordenados[0][0]:
        return colores_ordenados[0][1]
    
    # Si la posición está después del último color, retornar el último
    if pos >= colores_ordenados[-1][0]:
        return colores_ordenados[-1][1]
    
    # Buscar el intervalo correcto para interpolar
    for i in range(len(colores_ordenados) - 1):
        pos1, color1_str = colores_ordenados[i]
        pos2, color2_str = colores_ordenados[i + 1]
        
        if pos1 <= pos <= pos2:
            try:
                # Parsear colores RGB de manera segura
                color1 = parse_rgb_color(color1_str)
                color2 = parse_rgb_color(color2_str)
                
                # Calcular proporción de interpolación
                if pos2 - pos1 == 0:  # Evitar división por cero
                    prop = 0
                else:
                    prop = (pos - pos1) / (pos2 - pos1)
                
                # Interpolar cada componente RGB
                r = int(color1[0] + (color2[0] - color1[0]) * prop)
                g = int(color1[1] + (color2[1] - color1[1]) * prop)
                b = int(color1[2] + (color2[2] - color1[2]) * prop)
                
                return f'rgb({r}, {g}, {b})'
                
            except ValueError as e:
                raise ValueError(f"Error al parsear colores en posición {pos}: {e}")
    
    # Fallback (no debería llegar aquí)
    return colores_ordenados[-1][1]

# Definir la escala de colores base
COLOR_SCALE = [
    [0.0, 'rgb(240, 242, 246)'],    # Gris muy claro con tono azulado
    [1.0, 'rgb(26, 32, 44)']        # Gris muy oscuro/negro azulado
]

def generate_color_palette(n_colors, color_scale=None):
    """
    Genera una paleta de colores con n colores interpolados.
    
    Args:
        n_colors: Número de colores a generar
        color_scale: Escala de colores en formato [[posición, 'rgb(r,g,b)'], ...]
                    Ejemplo: [[0, 'rgb(240, 242, 246)'], [1, 'rgb(26, 32, 44)']]
                    Si es None, usa COLOR_SCALE por defecto
    
    Returns:
        Lista de strings con colores en formato 'rgb(r, g, b)'
    """
    if n_colors < 1:
        raise ValueError("n_colors debe ser al menos 1")
    
    if color_scale is None:
        color_scale = COLOR_SCALE
    
    if n_colors == 1:
        return [interpolar_color(0.5, color_scale)]
    
    # Generar posiciones equidistantes
    positions = np.linspace(0, 1, n_colors)
    
    # Interpolar colores en cada posición
    palette = []
    for pos in positions:
        color = interpolar_color(pos, color_scale)
        palette.append(color)
    
    return palette

def generate_color_palette_hex(n_colors, color_scale=None):
    """
    Genera una paleta de colores en formato hexadecimal.
    
    Args:
        n_colors: Número de colores a generar
        color_scale: Escala de colores en formato [[posición, 'rgb(r,g,b)'], ...]
                    Si es None, usa COLOR_SCALE por defecto
    
    Returns:
        Lista de strings con colores en formato hexadecimal '#rrggbb'
    """
    rgb_palette = generate_color_palette(n_colors, color_scale)
    hex_palette = []
    
    for rgb_color in rgb_palette:
        r, g, b = parse_rgb_color(rgb_color)
        hex_palette.append(rgb_to_hex(r, g, b))
    
    return hex_palette

def generate_plotly_colorscale(n_steps=20, color_scale=None):
    """
    Genera una escala de colores compatible con Plotly.
    
    Args:
        n_steps: Número de pasos en la escala
        color_scale: Escala de colores en formato [[posición, 'rgb(r,g,b)'], ...]
                    Si es None, usa COLOR_SCALE por defecto
    
    Returns:
        Lista de [posición, color] compatible con colorscale de Plotly
    """
    if color_scale is None:
        color_scale = COLOR_SCALE
    
    steps = np.linspace(0, 1, n_steps)
    plotly_scale = []
    
    for step in steps:
        color = interpolar_color(step, color_scale)
        plotly_scale.append([step, color])
    
    return plotly_scale

# Template base (mantenido igual que el original)
template_graficos = go.layout.Template()
template_graficos.layout = go.Layout(
    font=dict(
        family="Baskervville, monospace",
        size=14,
        color="black"
    ),
    title=dict(
        font=dict(
            family="Baskervville, monospace",
            size=16,
            color="black"
        ),
        x=0.5,
        y=0.95,
        xanchor='center',
        yanchor='top'
    ),
    xaxis=dict(
        showgrid=False,
        zeroline=True,
        title_font=dict(
            family="Baskervville, monospace",
            size=16,
            color="black"
        ),
        tickfont=dict(
            family="Baskervville, monospace",
            size=12,
            color="black"
        )
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=True,
        title_font=dict(
            family="Baskervville, monospace",
            size=16,
            color="black"
        ),
        tickfont=dict(
            family="Baskervville, monospace",
            size=12,
            color="black"
        )
    ),
    legend=dict(
        font=dict(
            family="Baskervville, monospace",
            size=12,
            color="black"
        ),
        bgcolor='white'
    ),
    width=700,
    height=500,
    plot_bgcolor='white',
    paper_bgcolor='white'
)


def show_visualizaciones():
    """Sección de visualizaciones con escalas de color"""
    st.header("Visualizaciones")
    st.write("Diferentes tipos de gráficos, visualizaciones y herramientas de diseño")
    
    # Tabs para organizar las visualizaciones
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Gráficos Básicos", 
        "Gráficos Avanzados", 
        "Mapas", 
        "Escalas de Color",
        "Dashboard Ejemplo"
    ])
    
    with tab1:
        st.subheader("Gráficos Básicos")
        st.write("Explora los diferentes tipos de gráficos básicos con datos de ejemplo")
        
        # Generar datos de ejemplo para los gráficos
        @st.cache_data
        def generate_chart_data():
            # Datos para gráficos de series temporales
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            time_series = pd.DataFrame({
                'fecha': dates,
                'ventas': np.cumsum(np.random.randn(100)) + 100,
                'gastos': np.cumsum(np.random.randn(100)) + 80,
                'beneficio': np.random.randn(100) * 10 + 20
            })
            
            # Datos categóricos
            categorias = pd.DataFrame({
                'categoria': ['A', 'B', 'C', 'D', 'E'],
                'valor': [23, 45, 56, 78, 32],
                'otro_valor': [12, 34, 23, 67, 89]
            })
            
            # Datos para scatter
            scatter_data = pd.DataFrame({
                'x': np.random.randn(200),
                'y': np.random.randn(200),
                'categoria': np.random.choice(['Tipo 1', 'Tipo 2', 'Tipo 3'], 200),
                'tamaño': np.random.randint(10, 100, 200)
            })
            
            # Datos para histograma
            hist_data = pd.DataFrame({
                'valores': np.random.normal(50, 15, 1000),
                'grupo': np.random.choice(['Grupo A', 'Grupo B'], 1000)
            })
            
            return time_series, categorias, scatter_data, hist_data
        
        time_series, categorias, scatter_data, hist_data = generate_chart_data()
        
        # Crear subtabs para diferentes tipos de gráficos
        subtab1, subtab2, subtab3, subtab4, subtab5, subtab6 = st.tabs([
            "Líneas", "Barras", "Scatter", "Pie", "Histogramas", "Box Plots"
        ])
        
        with subtab1:
            st.markdown("### Gráficos de Líneas")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Opciones:**")
                line_columns = st.multiselect(
                    "Selecciona variables:",
                    ['ventas', 'gastos', 'beneficio'],
                    default=['ventas', 'gastos'],
                    key="line_columns"
                )
                
                show_points = st.checkbox("Mostrar puntos", False, key="line_show_points")
                line_style = st.selectbox("Estilo de línea:", ["solid", "dash", "dot", "dashdot"], key="line_style")
            
            with col1:
                if line_columns:
                    fig = go.Figure()
                    
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                    for i, col in enumerate(line_columns):
                        mode = 'lines+markers' if show_points else 'lines'
                        fig.add_trace(go.Scatter(
                            x=time_series['fecha'],
                            y=time_series[col],
                            mode=mode,
                            name=col.title(),
                            line=dict(dash=line_style, color=colors[i % len(colors)])
                        ))
                    
                    fig.update_layout(
                        title="Serie Temporal - Datos Financieros",
                        xaxis_title="Fecha",
                        yaxis_title="Valor",
                        template=template_graficos,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Selecciona al menos una variable para mostrar el gráfico")
            
            # Código de ejemplo
            with st.expander("Ver código"):
                st.code('''
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['fecha'],
    y=df['ventas'],
    mode='lines',
    name='Ventas'
))

fig.update_layout(
    title="Gráfico de Líneas",
    xaxis_title="Fecha",
    yaxis_title="Valor"
)

st.plotly_chart(fig)
                ''', language='python')
        
        with subtab2:
            st.markdown("### Gráficos de Barras")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Opciones:**")
                bar_orientation = st.radio("Orientación:", ["Vertical", "Horizontal"], key="bar_orientation")
                bar_color = st.color_picker("Color de barras:", "#1f77b4", key="bar_color")
                show_values = st.checkbox("Mostrar valores", True, key="bar_show_values")
                chart_type = st.selectbox("Tipo:", ["Barras simples", "Barras agrupadas", "Barras apiladas"], key="bar_chart_type")
            
            with col1:
                if chart_type == "Barras simples":
                    fig = go.Figure()
                    
                    if bar_orientation == "Vertical":
                        fig.add_trace(go.Bar(
                            x=categorias['categoria'],
                            y=categorias['valor'],
                            marker_color=bar_color,
                            text=categorias['valor'] if show_values else None,
                            textposition='auto'
                        ))
                    else:
                        fig.add_trace(go.Bar(
                            x=categorias['valor'],
                            y=categorias['categoria'],
                            orientation='h',
                            marker_color=bar_color,
                            text=categorias['valor'] if show_values else None,
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title="Gráfico de Barras Simple",
                        template=template_graficos,
                        height=400
                    )
                
                elif chart_type == "Barras agrupadas":
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=categorias['categoria'],
                        y=categorias['valor'],
                        name='Valor 1',
                        marker_color='#1f77b4'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=categorias['categoria'],
                        y=categorias['otro_valor'],
                        name='Valor 2',
                        marker_color='#ff7f0e'
                    ))
                    
                    fig.update_layout(
                        title="Gráfico de Barras Agrupadas",
                        barmode='group',
                        template=template_graficos,
                        height=400
                    )
                
                else:  # Barras apiladas
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=categorias['categoria'],
                        y=categorias['valor'],
                        name='Valor 1',
                        marker_color='#1f77b4'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=categorias['categoria'],
                        y=categorias['otro_valor'],
                        name='Valor 2',
                        marker_color='#ff7f0e'
                    ))
                    
                    fig.update_layout(
                        title="Gráfico de Barras Apiladas",
                        barmode='stack',
                        template=template_graficos,
                        height=400
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Ver código"):
                st.code('''
# Barras simples
fig = go.Figure(data=[
    go.Bar(x=df['categoria'], y=df['valor'])
])

# Barras agrupadas
fig = go.Figure(data=[
    go.Bar(name='Serie 1', x=df['categoria'], y=df['valor1']),
    go.Bar(name='Serie 2', x=df['categoria'], y=df['valor2'])
])
fig.update_layout(barmode='group')

# Barras apiladas
fig.update_layout(barmode='stack')
                ''', language='python')
        
        with subtab3:
            st.markdown("### Gráficos de Dispersión (Scatter)")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Opciones:**")
                x_var = st.selectbox("Variable X:", ['x', 'y'], index=0, key="scatter_x_var")
                y_var = st.selectbox("Variable Y:", ['x', 'y'], index=1, key="scatter_y_var")
                color_var = st.selectbox("Color por:", [None, 'categoria'], index=1, key="scatter_color_var")
                size_var = st.selectbox("Tamaño por:", [None, 'tamaño'], index=1, key="scatter_size_var")
                opacity = st.slider("Opacidad:", 0.1, 1.0, 0.7, key="scatter_opacity")
            
            with col1:
                fig = go.Figure()
                
                if color_var and size_var:
                    # Scatter con color y tamaño
                    for cat in scatter_data[color_var].unique():
                        cat_data = scatter_data[scatter_data[color_var] == cat]
                        fig.add_trace(go.Scatter(
                            x=cat_data[x_var],
                            y=cat_data[y_var],
                            mode='markers',
                            name=cat,
                            marker=dict(
                                size=cat_data[size_var]/5,
                                opacity=opacity
                            )
                        ))
                elif color_var:
                    # Solo color
                    fig = px.scatter(
                        scatter_data, 
                        x=x_var, 
                        y=y_var, 
                        color=color_var,
                        opacity=opacity,
                        title="Gráfico de Dispersión"
                    )
                elif size_var:
                    # Solo tamaño
                    fig.add_trace(go.Scatter(
                        x=scatter_data[x_var],
                        y=scatter_data[y_var],
                        mode='markers',
                        marker=dict(
                            size=scatter_data[size_var]/5,
                            opacity=opacity
                        )
                    ))
                else:
                    # Scatter simple
                    fig.add_trace(go.Scatter(
                        x=scatter_data[x_var],
                        y=scatter_data[y_var],
                        mode='markers',
                        marker=dict(opacity=opacity)
                    ))
                
                fig.update_layout(
                    title="Gráfico de Dispersión",
                    xaxis_title=f"Variable {x_var.upper()}",
                    yaxis_title=f"Variable {y_var.upper()}",
                    template=template_graficos,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar correlación si es numérico
                if x_var != y_var:
                    correlation = scatter_data[x_var].corr(scatter_data[y_var])
                    st.metric("Correlación", f"{correlation:.3f}")
            
            with st.expander("Ver código"):
                st.code('''
# Scatter simple
fig = px.scatter(df, x='variable1', y='variable2')

# Scatter con color
fig = px.scatter(df, x='x', y='y', color='categoria')

# Scatter con tamaño
fig = px.scatter(df, x='x', y='y', size='tamaño')

# Scatter completo
fig = px.scatter(df, x='x', y='y', 
                color='categoria', size='tamaño',
                opacity=0.7)

st.plotly_chart(fig)
                ''', language='python')
        
        with subtab4:
            st.markdown("### Gráficos de Pie")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Opciones:**")
                pie_style = st.selectbox("Estilo:", ["Pie", "Donut"], key="pie_style")
                show_labels = st.checkbox("Mostrar etiquetas", True, key="pie_show_labels")
                show_values_pie = st.checkbox("Mostrar valores", True, key="pie_show_values")
                explode_slice = st.selectbox("Resaltar slice:", [None] + list(categorias['categoria']), key="pie_explode_slice")
            
            with col1:
                fig = go.Figure()
                
                # Preparar datos
                pull_values = [0.1 if cat == explode_slice else 0 for cat in categorias['categoria']]
                
                fig.add_trace(go.Pie(
                    labels=categorias['categoria'],
                    values=categorias['valor'],
                    hole=0.3 if pie_style == "Donut" else 0,
                    pull=pull_values,
                    textinfo='label+percent' if show_labels and show_values_pie else 
                             'label' if show_labels else 
                             'percent' if show_values_pie else 'none'
                ))
                
                fig.update_layout(
                    title="Gráfico de Pie",
                    template=template_graficos,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Ver código"):
                st.code('''
# Pie chart simple
fig = go.Figure(data=[go.Pie(
    labels=df['categoria'],
    values=df['valor']
)])

# Donut chart
fig = go.Figure(data=[go.Pie(
    labels=df['categoria'],
    values=df['valor'],
    hole=0.3
)])

# Con slice resaltado
fig = go.Figure(data=[go.Pie(
    labels=df['categoria'],
    values=df['valor'],
    pull=[0.1, 0, 0, 0, 0]  # Resalta primer slice
)])

st.plotly_chart(fig)
                ''', language='python')
        
        with subtab5:
            st.markdown("### Histogramas")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Opciones:**")
                bins = st.slider("Número de bins:", 10, 100, 30, key="hist_bins")
                overlay_group = st.checkbox("Superponer grupos", False, key="hist_overlay_group")
                show_curve = st.checkbox("Mostrar curva de densidad", False, key="hist_show_curve")
                hist_color = st.color_picker("Color:", "#1f77b4", key="hist_color")
            
            with col1:
                fig = go.Figure()
                
                if overlay_group:
                    # Histograma por grupos
                    for grupo in hist_data['grupo'].unique():
                        grupo_data = hist_data[hist_data['grupo'] == grupo]
                        fig.add_trace(go.Histogram(
                            x=grupo_data['valores'],
                            name=grupo,
                            nbinsx=bins,
                            opacity=0.7
                        ))
                    fig.update_layout(barmode='overlay')
                else:
                    # Histograma simple
                    fig.add_trace(go.Histogram(
                        x=hist_data['valores'],
                        nbinsx=bins,
                        marker_color=hist_color
                    ))
                
                # Agregar curva de densidad si se solicita
                if show_curve:
                    x_range = np.linspace(hist_data['valores'].min(), hist_data['valores'].max(), 100)
                    kde = stats.gaussian_kde(hist_data['valores'])
                    
                    # Escalar la curva para que se vea bien con el histograma
                    scale_factor = len(hist_data) * (hist_data['valores'].max() - hist_data['valores'].min()) / bins
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=kde(x_range) * scale_factor,
                        mode='lines',
                        name='Curva de densidad',
                        line=dict(color='red', width=2)
                    ))
                
                fig.update_layout(
                    title="Histograma",
                    xaxis_title="Valores",
                    yaxis_title="Frecuencia",
                    template=template_graficos,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estadísticas básicas
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Media", f"{hist_data['valores'].mean():.2f}")
                with col_b:
                    st.metric("Mediana", f"{hist_data['valores'].median():.2f}")
                with col_c:
                    st.metric("Desv. Est.", f"{hist_data['valores'].std():.2f}")
            
            with st.expander("Ver código"):
                st.code('''
# Histograma simple
fig = go.Figure(data=[go.Histogram(x=df['valores'])])

# Histograma con bins específicos
fig = go.Figure(data=[go.Histogram(
    x=df['valores'],
    nbinsx=50
)])

# Histograma por grupos
fig = px.histogram(df, x='valores', color='grupo', 
                  opacity=0.7, barmode='overlay')

st.plotly_chart(fig)
                ''', language='python')
        
        with subtab6:
            st.markdown("### Box Plots")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Opciones:**")
                box_orientation = st.radio("Orientación:", ["Vertical", "Horizontal"], key="box_orientation")
                show_points_box = st.selectbox("Mostrar puntos:", ["No", "Outliers", "Todos"], key="box_show_points")
                box_by_group = st.checkbox("Agrupar por categoría", True, key="box_by_group")
                notched = st.checkbox("Box plot con muescas", False, key="box_notched")
            
            with col1:
                fig = go.Figure()
                
                if box_by_group:
                    # Box plot por grupos
                    for grupo in hist_data['grupo'].unique():
                        grupo_data = hist_data[hist_data['grupo'] == grupo]
                        
                        if box_orientation == "Vertical":
                            fig.add_trace(go.Box(
                                y=grupo_data['valores'],
                                name=grupo,
                                boxpoints='outliers' if show_points_box == "Outliers" else 
                                         'all' if show_points_box == "Todos" else False,
                                notched=notched
                            ))
                        else:
                            fig.add_trace(go.Box(
                                x=grupo_data['valores'],
                                name=grupo,
                                boxpoints='outliers' if show_points_box == "Outliers" else 
                                         'all' if show_points_box == "Todos" else False,
                                notched=notched
                            ))
                else:
                    # Box plot simple
                    if box_orientation == "Vertical":
                        fig.add_trace(go.Box(
                            y=hist_data['valores'],
                            name="Distribución",
                            boxpoints='outliers' if show_points_box == "Outliers" else 
                                     'all' if show_points_box == "Todos" else False,
                            notched=notched
                        ))
                    else:
                        fig.add_trace(go.Box(
                            x=hist_data['valores'],
                            name="Distribución",
                            boxpoints='outliers' if show_points_box == "Outliers" else 
                                     'all' if show_points_box == "Todos" else False,
                            notched=notched
                        ))
                
                fig.update_layout(
                    title="Box Plot",
                    template=template_graficos,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estadísticas del box plot
                st.markdown("**Estadísticas:**")
                q1 = hist_data['valores'].quantile(0.25)
                q2 = hist_data['valores'].quantile(0.5)
                q3 = hist_data['valores'].quantile(0.75)
                iqr = q3 - q1
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Q1", f"{q1:.2f}")
                with col_b:
                    st.metric("Mediana", f"{q2:.2f}")
                with col_c:
                    st.metric("Q3", f"{q3:.2f}")
                with col_d:
                    st.metric("IQR", f"{iqr:.2f}")
            
            with st.expander("Ver código"):
                st.code('''
# Box plot simple
fig = go.Figure(data=[go.Box(y=df['valores'])])

# Box plot por grupos
fig = go.Figure()
for grupo in df['grupo'].unique():
    grupo_data = df[df['grupo'] == grupo]
    fig.add_trace(go.Box(
        y=grupo_data['valores'],
        name=grupo
    ))

# Con outliers
fig = go.Figure(data=[go.Box(
    y=df['valores'],
    boxpoints='outliers'
)])

st.plotly_chart(fig)
                ''', language='python')
    
    with tab2:
        st.subheader("Gráficos Avanzados")
        st.write("Visualizaciones más complejas e interactivas")
        
        # Subtabs para gráficos avanzados
        adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5 = st.tabs([
            "3D", "Subplots", "Animaciones", "Heatmaps", "Radar"
        ])
        
        with adv_tab1:
            st.markdown("### Gráficos 3D")
            
            # Generar datos 3D
            @st.cache_data
            def generate_3d_data():
                n = 100
                x = np.random.randn(n)
                y = np.random.randn(n)
                z = x**2 + y**2 + np.random.randn(n) * 0.1
                
                # Superficie 3D
                x_surf = np.linspace(-3, 3, 30)
                y_surf = np.linspace(-3, 3, 30)
                X, Y = np.meshgrid(x_surf, y_surf)
                Z = np.sin(X) * np.cos(Y)
                
                return x, y, z, X, Y, Z
            
            x, y, z, X, Y, Z = generate_3d_data()
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Opciones 3D:**")
                plot_3d_type = st.selectbox("Tipo de gráfico:", 
                                          ["Scatter 3D", "Superficie", "Wireframe", "Contorno 3D"],
                                          key="plot_3d_type")
                color_scheme = st.selectbox("Esquema de color:", 
                                          ["Viridis", "Plasma", "Blues", "Reds"],
                                          key="3d_color_scheme")
            
            with col1:
                fig = go.Figure()
                
                if plot_3d_type == "Scatter 3D":
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=z,
                            colorscale=color_scheme,
                            showscale=True
                        )
                    ))
                    fig.update_layout(title="Scatter Plot 3D")
                
                elif plot_3d_type == "Superficie":
                    fig.add_trace(go.Surface(
                        x=X, y=Y, z=Z,
                        colorscale=color_scheme
                    ))
                    fig.update_layout(title="Gráfico de Superficie 3D")
                
                elif plot_3d_type == "Wireframe":
                    fig.add_trace(go.Surface(
                        x=X, y=Y, z=Z,
                        colorscale=color_scheme,
                        showscale=False,
                        opacity=0.8,
                        surfacecolor=np.zeros_like(Z)
                    ))
                    fig.update_layout(title="Wireframe 3D")
                
                else:  # Contorno 3D
                    fig.add_trace(go.Surface(
                        x=X, y=Y, z=Z,
                        colorscale=color_scheme,
                        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                    ))
                    fig.update_layout(title="Contorno 3D")
                
                fig.update_layout(
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
                    ),
                    template=template_graficos,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Ver código 3D"):
                st.code('''
# Scatter 3D
fig = go.Figure(data=[go.Scatter3d(
    x=df['x'], y=df['y'], z=df['z'],
    mode='markers',
    marker=dict(color=df['z'], colorscale='Viridis')
)])

# Superficie 3D
fig = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')])

# Configurar escena 3D
fig.update_layout(scene=dict(
    xaxis_title="X",
    yaxis_title="Y", 
    zaxis_title="Z"
))

st.plotly_chart(fig)
                ''', language='python')
        
        with adv_tab2:
            st.markdown("### Subplots")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Configuración:**")
                subplot_rows = st.selectbox("Filas:", [1, 2, 3], index=1, key="subplot_rows")
                subplot_cols = st.selectbox("Columnas:", [1, 2, 3], index=1, key="subplot_cols")
                shared_axes = st.checkbox("Ejes compartidos", False, key="subplot_shared_axes")
            
            with col1:
                
                # Regenerar algunos datos para subplots
                time_series, categorias, scatter_data, hist_data = generate_chart_data()
                
                fig = make_subplots(
                    rows=subplot_rows, 
                    cols=subplot_cols,
                    shared_xaxes=shared_axes,
                    shared_yaxes=shared_axes,
                    subplot_titles=("Serie Temporal", "Barras", "Scatter", "Histograma")[:subplot_rows*subplot_cols]
                )
                
                # Agregar gráficos según el número de subplots
                plots_to_add = []
                
                # Plot 1: Serie temporal
                plots_to_add.append(go.Scatter(
                    x=time_series['fecha'], 
                    y=time_series['ventas'],
                    mode='lines',
                    name='Ventas'
                ))
                
                # Plot 2: Barras
                if subplot_rows * subplot_cols > 1:
                    plots_to_add.append(go.Bar(
                        x=categorias['categoria'],
                        y=categorias['valor'],
                        name='Categorías'
                    ))
                
                # Plot 3: Scatter
                if subplot_rows * subplot_cols > 2:
                    plots_to_add.append(go.Scatter(
                        x=scatter_data['x'],
                        y=scatter_data['y'],
                        mode='markers',
                        name='Scatter'
                    ))
                
                # Plot 4: Histograma
                if subplot_rows * subplot_cols > 3:
                    plots_to_add.append(go.Histogram(
                        x=hist_data['valores'],
                        name='Distribución'
                    ))
                
                # Agregar plots a subplots
                plot_idx = 0
                for row in range(1, subplot_rows + 1):
                    for col in range(1, subplot_cols + 1):
                        if plot_idx < len(plots_to_add):
                            fig.add_trace(plots_to_add[plot_idx], row=row, col=col)
                            plot_idx += 1
                
                fig.update_layout(
                    title="Dashboard con Subplots",
                    template=template_graficos,
                    height=400 * subplot_rows,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Ver código Subplots"):
                st.code('''
from plotly.subplots import make_subplots

# Crear subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4")
)

# Agregar gráficos
fig.add_trace(go.Scatter(x=df['x'], y=df['y']), row=1, col=1)
fig.add_trace(go.Bar(x=df['cat'], y=df['val']), row=1, col=2)
fig.add_trace(go.Histogram(x=df['values']), row=2, col=1)
fig.add_trace(go.Box(y=df['data']), row=2, col=2)

fig.update_layout(title="Dashboard")
st.plotly_chart(fig)
                ''', language='python')
        
        with adv_tab3:
            st.markdown("### Gráficos Animados")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Animación:**")
                animation_type = st.selectbox("Tipo:", ["Scatter animado", "Barras animadas", "Línea temporal"], key="animation_type")
                frame_duration = st.slider("Duración frame (ms):", 100, 2000, 500, key="frame_duration")
                show_animation = st.checkbox("Reproducir automáticamente", True, key="show_animation")
            
            with col1:
                # Generar datos para animación
                @st.cache_data
                def generate_animation_data():
                    frames = []
                    for i in range(20):
                        frame_data = pd.DataFrame({
                            'x': np.random.randn(50) + i*0.1,
                            'y': np.random.randn(50) + i*0.1,
                            'size': np.random.randint(10, 50, 50),
                            'frame': i
                        })
                        frames.append(frame_data)
                    return pd.concat(frames)
                
                anim_data = generate_animation_data()
                
                if animation_type == "Scatter animado":
                    fig = px.scatter(
                        anim_data, 
                        x="x", y="y", 
                        size="size",
                        animation_frame="frame",
                        range_x=[-3, 5], 
                        range_y=[-3, 5],
                        title="Scatter Plot Animado"
                    )
                
                elif animation_type == "Barras animadas":
                    # Datos para barras animadas
                    bar_frames = []
                    categories = ['A', 'B', 'C', 'D', 'E']
                    for i in range(10):
                        frame_data = pd.DataFrame({
                            'categoria': categories,
                            'valor': np.random.randint(10, 100, 5),
                            'frame': i
                        })
                        bar_frames.append(frame_data)
                    bar_data = pd.concat(bar_frames)
                    
                    fig = px.bar(
                        bar_data,
                        x="categoria", y="valor",
                        animation_frame="frame",
                        range_y=[0, 100],
                        title="Gráfico de Barras Animado"
                    )
                
                else:  # Línea temporal
                    # Crear datos de línea temporal
                    dates = pd.date_range('2023-01-01', periods=50, freq='D')
                    line_frames = []
                    cumulative_value = 0
                    
                    for i in range(len(dates)):
                        cumulative_value += np.random.randn() * 2
                        frame_data = pd.DataFrame({
                            'fecha': dates[:i+1],
                            'valor': np.cumsum(np.random.randn(i+1)) + cumulative_value,
                            'frame': [i] * (i+1)
                        })
                        line_frames.append(frame_data)
                    
                    # Usar solo algunos frames para evitar demasiados datos
                    selected_frames = line_frames[::5]  # Cada 5 frames
                    line_data = pd.concat(selected_frames)
                    
                    fig = px.line(
                        line_data,
                        x="fecha", y="valor",
                        animation_frame="frame",
                        title="Línea Temporal Animada"
                    )
                
                # Configurar animación
                fig.update_layout(
                    template=template_graficos,
                    height=400,
                    updatemenus=[{
                        "buttons": [
                            {"args": [None, {"frame": {"duration": frame_duration, "redraw": True}, 
                                           "fromcurrent": True}], "label": "Play", "method": "animate"},
                            {"args": [[None], {"frame": {"duration": 0, "redraw": True}, 
                                             "mode": "immediate", "transition": {"duration": 0}}], 
                             "label": "Pause", "method": "animate"}
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 87},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0,
                        "yanchor": "top"
                    }]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Ver código Animación"):
                st.code('''
# Scatter animado
fig = px.scatter(df, x="x", y="y", 
                animation_frame="time",
                size="size", color="category")

# Barras animadas  
fig = px.bar(df, x="category", y="value",
            animation_frame="year")

# Línea animada
fig = px.line(df, x="date", y="value",
             animation_frame="frame")

# Configurar velocidad
fig.update_layout(
    updatemenus=[{
        "buttons": [
            {"args": [None, {"frame": {"duration": 500}}], 
             "label": "Play", "method": "animate"}
        ]
    }]
)

st.plotly_chart(fig)
                ''', language='python')
        
        with adv_tab4:
            st.markdown("### Heatmaps Avanzados")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Configuración:**")
                heatmap_type = st.selectbox("Tipo:", ["Correlación", "Matriz de datos", "Calendario"], key="heatmap_type")
                colorscale_heat = st.selectbox("Escala de color:", 
                                             ["RdBu", "Viridis", "Blues", "Reds", "RdYlBu"],
                                             key="colorscale_heat")
                show_text_heat = st.checkbox("Mostrar valores", True, key="show_text_heat")
            
            with col1:
                if heatmap_type == "Correlación":
                    # Matriz de correlación
                    corr_data = pd.DataFrame({
                        'Variable_A': np.random.randn(100),
                        'Variable_B': np.random.randn(100),
                        'Variable_C': np.random.randn(100),
                        'Variable_D': np.random.randn(100)
                    })
                    # Agregar algunas correlaciones
                    corr_data['Variable_B'] = corr_data['Variable_A'] * 0.7 + np.random.randn(100) * 0.3
                    corr_data['Variable_C'] = corr_data['Variable_A'] * -0.5 + np.random.randn(100) * 0.5
                    
                    corr_matrix = corr_data.corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale=colorscale_heat,
                        text=corr_matrix.round(2).values if show_text_heat else None,
                        texttemplate="%{text}" if show_text_heat else None,
                        textfont={"size": 12},
                        zmin=-1, zmax=1
                    ))
                    
                    fig.update_layout(
                        title="Matriz de Correlación",
                        template=template_graficos,
                        height=400
                    )
                
                elif heatmap_type == "Matriz de datos":
                    # Matriz de datos aleatoria
                    data_matrix = np.random.randn(10, 15)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=data_matrix,
                        colorscale=colorscale_heat,
                        text=data_matrix.round(1) if show_text_heat else None,
                        texttemplate="%{text}" if show_text_heat else None
                    ))
                    
                    fig.update_layout(
                        title="Heatmap de Datos",
                        template=template_graficos,
                        height=400
                    )
                
                else:  # Calendario heatmap
                    # Generar datos de calendario
                    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
                    calendar_data = pd.DataFrame({
                        'date': dates,
                        'value': np.random.randint(0, 100, len(dates))
                    })
                    
                    # Crear matriz tipo calendario
                    calendar_data['week'] = calendar_data['date'].dt.isocalendar().week
                    calendar_data['dayofweek'] = calendar_data['date'].dt.dayofweek
                    
                    # Pivot para crear matriz
                    calendar_matrix = calendar_data.pivot(index='dayofweek', columns='week', values='value')
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=calendar_matrix.values,
                        x=[f"Semana {i}" for i in calendar_matrix.columns],
                        y=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
                        colorscale=colorscale_heat,
                        showscale=True
                    ))
                    
                    fig.update_layout(
                        title="Heatmap tipo Calendario 2023",
                        template=template_graficos,
                        height=300,
                        xaxis_title="Semanas",
                        yaxis_title="Días de la semana"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Ver código Heatmap"):
                st.code('''
# Heatmap de correlación
corr_matrix = df.corr()
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    text=corr_matrix.round(2),
    texttemplate="%{text}"
))

# Heatmap simple
fig = go.Figure(data=go.Heatmap(
    z=data_matrix,
    colorscale='Viridis'
))

# Con anotaciones
fig = px.imshow(data_matrix, 
               text_auto=True,
               aspect="auto")

st.plotly_chart(fig)
                ''', language='python')
        
        with adv_tab5:
            st.markdown("### Gráficos de Radar")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Configuración:**")
                radar_mode = st.selectbox("Modo:", ["Simple", "Múltiples series", "Comparación"], key="radar_mode")
                fill_area = st.checkbox("Rellenar área", True, key="radar_fill_area")
                show_markers_radar = st.checkbox("Mostrar marcadores", True, key="radar_show_markers")
            
            with col1:
                # Datos para radar
                categories = ['Velocidad', 'Potencia', 'Resistencia', 'Agilidad', 'Técnica', 'Estrategia']
                
                if radar_mode == "Simple":
                    values = [4, 3, 5, 4, 2, 4]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],  # Cerrar el polígono
                        theta=categories + [categories[0]],
                        fill='toself' if fill_area else None,
                        name='Jugador 1',
                        line_color='blue',
                        marker=dict(size=8) if show_markers_radar else None
                    ))
                
                elif radar_mode == "Múltiples series":
                    player1 = [4, 3, 5, 4, 2, 4]
                    player2 = [3, 5, 3, 5, 4, 3]
                    player3 = [5, 2, 4, 3, 5, 5]
                    
                    fig = go.Figure()
                    
                    colors = ['blue', 'red', 'green']
                    players = [player1, player2, player3]
                    names = ['Jugador 1', 'Jugador 2', 'Jugador 3']
                    
                    for i, (player, name, color) in enumerate(zip(players, names, colors)):
                        fig.add_trace(go.Scatterpolar(
                            r=player + [player[0]],
                            theta=categories + [categories[0]],
                            fill='toself' if fill_area else None,
                            name=name,
                            line_color=color,
                            opacity=0.7,
                            marker=dict(size=6) if show_markers_radar else None
                        ))
                
                else:  # Comparación
                    antes = [2, 3, 2, 4, 2, 3]
                    despues = [4, 4, 5, 5, 4, 4]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=antes + [antes[0]],
                        theta=categories + [categories[0]],
                        fill='toself' if fill_area else None,
                        name='Antes',
                        line_color='red',
                        opacity=0.6
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=despues + [despues[0]],
                        theta=categories + [categories[0]],
                        fill='toself' if fill_area else None,
                        name='Después',
                        line_color='green',
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5]
                        )
                    ),
                    showlegend=True,
                    title="Gráfico de Radar - Análisis de Rendimiento",
                    template=template_graficos,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar datos en tabla
                if radar_mode == "Simple":
                    data_df = pd.DataFrame({
                        'Categoría': categories,
                        'Valor': values
                    })
                    st.dataframe(data_df, use_container_width=True, hide_index=True)
            
            with st.expander("Ver código Radar"):
                st.code('''
# Gráfico de radar simple
categories = ['A', 'B', 'C', 'D', 'E']
values = [4, 3, 5, 4, 2]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=values + [values[0]],  # Cerrar polígono
    theta=categories + [categories[0]],
    fill='toself',
    name='Serie 1'
))

# Múltiples series
for i, serie in enumerate(series_data):
    fig.add_trace(go.Scatterpolar(
        r=serie + [serie[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=f'Serie {i+1}',
        opacity=0.7
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    showlegend=True
)

st.plotly_chart(fig)
                ''', language='python')
    
    with tab3:
        st.subheader("🗺️ Mapas")
        st.write("Visualizaciones geográficas y mapas interactivos")
        
        # Subtabs para diferentes tipos de mapas
        map_tab1, map_tab2= st.tabs(["Mapas de Puntos", "Mapas de Calor"])
        
        with map_tab1:
            st.markdown("### Mapas de Puntos")
            
            # Generar datos de ejemplo para mapas
            @st.cache_data
            def generate_map_data():
                # Coordenadas de ciudades principales de España
                cities_data = pd.DataFrame({
                    'ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Zaragoza', 
                             'Málaga', 'Murcia', 'Palma', 'Las Palmas', 'Bilbao'],
                    'lat': [40.4168, 41.3851, 39.4699, 37.3891, 41.6488, 
                           36.7213, 37.9922, 39.5696, 28.1248, 43.2627],
                    'lon': [-3.7038, 2.1734, -0.3763, -5.9845, -0.8891, 
                           -4.4214, -1.1307, 2.6502, -15.4300, -2.9253],
                    'poblacion': [6.6, 5.6, 2.5, 1.9, 1.3, 1.6, 1.5, 0.9, 0.8, 1.1],
                    'ventas': np.random.randint(100, 1000, 10)
                })
                return cities_data
            
            cities_data = generate_map_data()
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Configuración:**")
                size_by = st.selectbox("Tamaño por:", ['poblacion', 'ventas', 'Fijo'], key="map_size_by")
                color_by = st.selectbox("Color por:", ['poblacion', 'ventas', 'ciudad'], key="map_color_by")
                map_style = st.selectbox("Estilo:", ['open-street-map', 'carto-positron', 'stamen-terrain'], key="map_style")
                zoom_level = st.slider("Zoom:", 3, 10, 5, key="map_zoom_level")
            
            with col1:
                fig = go.Figure()
                
                if size_by == 'Fijo':
                    size_values = [10] * len(cities_data)
                else:
                    size_values = cities_data[size_by]
                
                if color_by in ['poblacion', 'ventas']:
                    color_values = cities_data[color_by]
                    colorscale = 'Viridis'
                else:
                    color_values = cities_data['ciudad']
                    colorscale = None
                
                fig.add_trace(go.Scattermapbox(
                    lat=cities_data['lat'],
                    lon=cities_data['lon'],
                    mode='markers',
                    marker=dict(
                        size=size_values * 2,  # Escalar para visibilidad
                        color=color_values,
                        colorscale=colorscale,
                        showscale=True if color_by in ['poblacion', 'ventas'] else False,
                        sizemode='diameter'
                    ),
                    text=cities_data['ciudad'],
                    hovertemplate='<b>%{text}</b><br>' +
                                'Población: %{customdata[0]:.1f}M<br>' +
                                'Ventas: %{customdata[1]}<br>' +
                                '<extra></extra>',
                    customdata=np.column_stack((cities_data['poblacion'], cities_data['ventas']))
                ))
                
                fig.update_layout(
                    mapbox_style=map_style,
                    mapbox=dict(
                        accesstoken=None,  # Para mapas públicos no se necesita token
                        bearing=0,
                        center=dict(lat=40.0, lon=-4.0),  # Centro en España
                        pitch=0,
                        zoom=zoom_level
                    ),
                    height=500,
                    margin={"r":0,"t":0,"l":0,"b":0},
                    title="Mapa de Ciudades Españolas"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar datos
                st.dataframe(cities_data, use_container_width=True, hide_index=True)
        
        with map_tab2:
            st.markdown("### Mapas de Calor (Heatmaps Geográficos)")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("**Configuración:**")
                density_var = st.selectbox("Variable de densidad:", ['poblacion', 'ventas'], key="map_density_var")
                radius_size = st.slider("Radio de influencia:", 10, 50, 25, key="map_radius_size")
                opacity_level = st.slider("Opacidad:", 0.1, 1.0, 0.7, key="map_opacity_level")
            
            with col1:
                # Para mapas de calor, usamos scatter con alta densidad
                fig = go.Figure()
                
                # Expandir datos para crear efecto de mapa de calor
                expanded_data = []
                for _, row in cities_data.iterrows():
                    # Generar puntos alrededor de cada ciudad
                    n_points = int(row[density_var] * 10)  # Más puntos = más densidad
                    for _ in range(n_points):
                        # Añadir ruido gaussiano para dispersar puntos
                        lat_noise = np.random.normal(0, 0.1)
                        lon_noise = np.random.normal(0, 0.1)
                        expanded_data.append({
                            'lat': row['lat'] + lat_noise,
                            'lon': row['lon'] + lon_noise,
                            'ciudad': row['ciudad'],
                            'valor': row[density_var]
                        })
                
                expanded_df = pd.DataFrame(expanded_data)
                
                fig.add_trace(go.Scattermapbox(
                    lat=expanded_df['lat'],
                    lon=expanded_df['lon'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=expanded_df['valor'],
                        colorscale='Hot',
                        opacity=opacity_level,
                        showscale=True
                    ),
                    text=expanded_df['ciudad'],
                    hovertemplate='<b>%{text}</b><br>Valor: %{marker.color}<extra></extra>'
                ))
                
                fig.update_layout(
                    mapbox_style='carto-positron',
                    mapbox=dict(
                        center=dict(lat=40.0, lon=-4.0),
                        zoom=5
                    ),
                    height=500,
                    margin={"r":0,"t":0,"l":0,"b":0},
                    title=f"Mapa de Calor - {density_var.title()}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Los mapas de calor muestran la densidad de datos. Áreas más rojas indican mayor concentración.")
        

    with tab4:
        # Esta tab ya está implementada (Escalas de Color)
        st.subheader("Generador de Escalas de Color")
        st.write("Crea escalas de color personalizadas para tus visualizaciones")
        
        # Configuración de la escala
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Configuración de la Escala")
            
            # Número de colores en la escala
            num_colors = st.slider(
                "Número de colores en la escala:",
                min_value=2,
                max_value=10,
                value=3,
                help="Define cuántos colores quieres usar para crear tu escala",
                key="color_scale_num_colors"
            )
            
            # Inicializar session state para los colores si no existe
            if 'scale_colors' not in st.session_state:
                st.session_state.scale_colors = {
                    0: "#F0F2F6",  # Gris claro
                    1: "#1A202C"   # Gris oscuro
                }
            
            # Ajustar el diccionario de colores según el número seleccionado
            current_num = len(st.session_state.scale_colors)
            if num_colors < current_num:
                # Eliminar colores extras
                for i in range(num_colors, current_num):
                    if i in st.session_state.scale_colors:
                        del st.session_state.scale_colors[i]
            elif num_colors > current_num:
                # Añadir nuevos colores
                for i in range(current_num, num_colors):
                    # Interpolar un color inicial para la nueva posición
                    if i not in st.session_state.scale_colors:
                        st.session_state.scale_colors[i] = "#808080"  # Gris medio por defecto
        
        with col2:
            st.markdown("### Información")
            st.info(
                "Selecciona los colores de tu escala. Los colores intermedios "
                "se interpolarán automáticamente."
            )
            
            # Botón para resetear a valores por defecto
            if st.button("Resetear a valores por defecto", key="reset_color_scale"):
                st.session_state.scale_colors = {
                    0: "#F0F2F6",
                    1: "#1A202C"
                }
                st.rerun()
        
        # Selector de colores
        st.markdown("---")
        st.markdown("### Selección de Colores")
        
        # Crear columnas dinámicas para los color pickers
        cols = st.columns(min(num_colors, 5))  # Máximo 5 columnas
        
        color_positions = []
        for i in range(num_colors):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Calcular la posición del color en la escala (0 a 1)
                if num_colors == 1:
                    position = 0.5
                else:
                    position = i / (num_colors - 1)
                
                # Color picker
                color = st.color_picker(
                    f"Color {i+1} (pos: {position:.2f})",
                    value=st.session_state.scale_colors.get(i, "#808080"),
                    key=f"color_scale_picker_{i}"
                )
                
                # Actualizar el color en session state
                st.session_state.scale_colors[i] = color
                
                # Convertir hex a rgb
                hex_color = color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                rgb_str = f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
                
                color_positions.append([position, rgb_str])
        
        # Visualización de la escala
        st.markdown("---")
        st.markdown("### Vista Previa de la Escala")
        
        # Generar la visualización usando plotly
        preview_steps = st.slider("Pasos de interpolación:", 10, 100, 50, key="color_scale_preview_steps")
        
        # Crear la escala interpolada
        interpolated_colors = []
        positions = np.linspace(0, 1, preview_steps)
        
        for pos in positions:
            color = interpolar_color(pos, color_positions)
            interpolated_colors.append(color)
        
        # Crear gráfico de vista previa
        fig = go.Figure()
        
        # Añadir barras de color
        for i, (pos, color) in enumerate(zip(positions, interpolated_colors)):
            fig.add_trace(go.Bar(
                x=[pos],
                y=[1],
                width=1/preview_steps,
                marker_color=color,
                showlegend=False,
                hovertemplate=f'Posición: {pos:.3f}<br>Color: {color}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Escala de Color Interpolada",
            xaxis_title="Posición",
            yaxis_visible=False,
            height=200,
            bargap=0,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Paletas de colores discretas
        st.markdown("---")
        st.markdown("### Paletas Discretas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            discrete_n = st.number_input(
                "Número de colores discretos:",
                min_value=2,
                max_value=20,
                value=5,
                key="color_scale_discrete_n"
            )
        
        with col2:
            format_type = st.selectbox(
                "Formato de salida:",
                ["RGB", "Hexadecimal"],
                key="color_scale_format_type"
            )
        
        # Generar paleta discreta
        if format_type == "RGB":
            discrete_palette = generate_color_palette(discrete_n, color_positions)
        else:
            discrete_palette = generate_color_palette_hex(discrete_n, color_positions)
        
        # Mostrar la paleta discreta
        st.write("**Paleta generada:**")
        
        # Visualizar colores discretos
        cols = st.columns(min(discrete_n, 10))
        for i, color in enumerate(discrete_palette):
            col_idx = i % len(cols)
            with cols[col_idx]:
                if format_type == "RGB":
                    # Convertir rgb string a hex para mostrar
                    match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
                    if match:
                        r, g, b = map(int, match.groups())
                        hex_color = f"#{r:02x}{g:02x}{b:02x}"
                        st.color_picker(f"", value=hex_color, disabled=True, key=f"color_scale_discrete_rgb_{i}")
                        st.caption(color)
                else:
                    st.color_picker(f"", value=color, disabled=True, key=f"color_scale_discrete_hex_{i}")
                    st.caption(color)
        
        # Código para copiar
        st.markdown("---")
        st.markdown("### Código de la Escala")
        
        with st.expander("Ver código Python para usar esta escala"):
            code = f"""# Escala de colores personalizada
color_scale = {color_positions}

# Generar paleta discreta
palette = generate_color_palette({discrete_n}, color_scale)

# Para Plotly
plotly_scale = generate_plotly_colorscale(n_steps=20, color_scale=color_scale)
"""
            st.code(code, language='python')
        
        # Sección de ejemplos de visualización
        st.markdown("---")
        st.markdown("### Ejemplos de Visualización")
        
        # Generar datos de ejemplo
        example_data = pd.DataFrame({
            'categoría': [f'Cat {i+1}' for i in range(discrete_n)],
            'valor': np.random.randint(50, 200, discrete_n)
        })
        
        example_tabs = st.tabs(["Barras", "Pie", "Heatmap", "Scatter"])
        
        with example_tabs[0]:
            # Gráfico de barras
            fig = go.Figure(data=[
                go.Bar(
                    x=example_data['categoría'],
                    y=example_data['valor'],
                    marker_color=discrete_palette[:len(example_data)]
                )
            ])
            fig.update_layout(
                title="Gráfico de Barras con Escala Personalizada",
                template=template_graficos
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with example_tabs[1]:
            # Gráfico de pie
            fig = go.Figure(data=[
                go.Pie(
                    labels=example_data['categoría'],
                    values=example_data['valor'],
                    marker=dict(colors=discrete_palette[:len(example_data)])
                )
            ])
            fig.update_layout(
                title="Gráfico de Pie con Escala Personalizada",
                template=template_graficos
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with example_tabs[2]:
            # Heatmap
            # Generar datos para heatmap
            heatmap_data = np.random.randn(10, 10)
            
            # Convertir la escala a formato Plotly
            plotly_colorscale = generate_plotly_colorscale(20, color_positions)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                colorscale=plotly_colorscale,
                colorbar=dict(title="Valor")
            ))
            fig.update_layout(
                title="Heatmap con Escala Personalizada",
                template=template_graficos
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with example_tabs[3]:
            # Scatter plot con gradiente
            scatter_data = pd.DataFrame({
                'x': np.random.randn(100),
                'y': np.random.randn(100),
                'valor': np.random.randn(100)
            })
            
            fig = go.Figure(data=go.Scatter(
                x=scatter_data['x'],
                y=scatter_data['y'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=scatter_data['valor'],
                    colorscale=plotly_colorscale,
                    showscale=True,
                    colorbar=dict(title="Valor")
                )
            ))
            fig.update_layout(
                title="Scatter Plot con Escala Personalizada",
                template=template_graficos
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Exportar escala
        st.markdown("---")
        st.markdown("### Exportar Escala")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Exportar como JSON
            export_data = {
                'color_scale': color_positions,
                'discrete_palette_rgb': generate_color_palette(discrete_n, color_positions),
                'discrete_palette_hex': generate_color_palette_hex(discrete_n, color_positions)
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Descargar como JSON",
                data=json_str,
                file_name="color_scale.json",
                mime="application/json",
                key="color_scale_download_json"
            )
        
        with col2:
            # Exportar como CSS
            css_vars = ":root {\n"
            for i, color in enumerate(generate_color_palette_hex(discrete_n, color_positions)):
                css_vars += f"  --color-{i+1}: {color};\n"
            css_vars += "}"
            
            st.download_button(
                label="Descargar variables CSS",
                data=css_vars,
                file_name="color_scale.css",
                mime="text/css",
                key="color_scale_download_css"
            )
        
        with col3:
            # Exportar como Python
            python_code = f"""# Escala de colores generada
from color_scale_functions import *

# Definición de la escala
COLOR_SCALE = {color_positions}

# Paleta RGB
RGB_PALETTE = {generate_color_palette(discrete_n, color_positions)}

# Paleta Hexadecimal
HEX_PALETTE = {generate_color_palette_hex(discrete_n, color_positions)}

# Escala para Plotly
PLOTLY_SCALE = generate_plotly_colorscale(20, COLOR_SCALE)
"""
            st.download_button(
                label="Descargar código Python",
                data=python_code,
                file_name="color_scale.py",
                mime="text/plain",
                key="color_scale_download_python"
            )
    
    with tab5:
        st.subheader("Dashboard Ejemplo")
        st.write("Dashboard completo integrando múltiples visualizaciones")
        
        # Generar datos completos para el dashboard
        @st.cache_data
        def generate_dashboard_data():
            # Datos de ventas mensuales
            months = pd.date_range('2023-01-01', periods=12, freq='M')
            sales_data = pd.DataFrame({
                'mes': months,
                'ventas': np.random.randint(1000, 5000, 12),
                'objetivo': [3000] * 12,
                'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 12)
            })
            
            # Datos de productos
            products_data = pd.DataFrame({
                'producto': ['Producto A', 'Producto B', 'Producto C', 'Producto D', 'Producto E'],
                'ventas': np.random.randint(500, 2000, 5),
                'margen': np.random.uniform(0.1, 0.4, 5),
                'categoria': ['Cat 1', 'Cat 1', 'Cat 2', 'Cat 2', 'Cat 3']
            })
            
            # Datos de satisfacción del cliente
            satisfaction_data = pd.DataFrame({
                'aspecto': ['Calidad', 'Precio', 'Servicio', 'Entrega', 'Soporte'],
                'puntuacion': [4.2, 3.8, 4.5, 4.0, 3.9]
            })
            
            return sales_data, products_data, satisfaction_data
        
        sales_data, products_data, satisfaction_data = generate_dashboard_data()
        
        # KPIs principales
        st.markdown("### KPIs Principales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = sales_data['ventas'].sum()
            st.metric("Ventas Totales", f"€{total_sales:,}", "12%")
        
        with col2:
            avg_satisfaction = satisfaction_data['puntuacion'].mean()
            st.metric("Satisfacción Promedio", f"{avg_satisfaction:.1f}/5", "0.2")
        
        with col3:
            best_product = products_data.loc[products_data['ventas'].idxmax(), 'producto']
            st.metric("Producto Top", best_product, "15%")
        
        with col4:
            avg_margin = products_data['margen'].mean()
            st.metric("Margen Promedio", f"{avg_margin:.1%}", "2%")
        
        st.markdown("---")
        
        # Gráficos principales del dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de ventas mensuales
            fig_sales = go.Figure()
            
            fig_sales.add_trace(go.Scatter(
                x=sales_data['mes'],
                y=sales_data['ventas'],
                mode='lines+markers',
                name='Ventas Reales',
                line=dict(color='#1f77b4', width=3)
            ))
            
            fig_sales.add_trace(go.Scatter(
                x=sales_data['mes'],
                y=sales_data['objetivo'],
                mode='lines',
                name='Objetivo',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig_sales.update_layout(
                title="Evolución de Ventas Mensuales",
                xaxis_title="Mes",
                yaxis_title="Ventas (€)",
                template=template_graficos,
                height=400
            )
            
            st.plotly_chart(fig_sales, use_container_width=True)
        
        with col2:
            # Gráfico de productos
            fig_products = go.Figure(data=[
                go.Bar(
                    x=products_data['producto'],
                    y=products_data['ventas'],
                    marker_color=products_data['ventas'],
                    marker_colorscale='Viridis',
                    text=products_data['ventas'],
                    textposition='auto'
                )
            ])
            
            fig_products.update_layout(
                title="Ventas por Producto",
                xaxis_title="Producto",
                yaxis_title="Ventas (€)",
                template=template_graficos,
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_products, use_container_width=True)
        
        # Segunda fila de gráficos
        col3, col4 = st.columns(2)
        
        with col3:
            # Gráfico de radar para satisfacción
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=satisfaction_data['puntuacion'].tolist() + [satisfaction_data['puntuacion'].iloc[0]],
                theta=satisfaction_data['aspecto'].tolist() + [satisfaction_data['aspecto'].iloc[0]],
                fill='toself',
                name='Satisfacción del Cliente',
                line_color='green',
                fillcolor='rgba(0, 255, 0, 0.3)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )
                ),
                showlegend=False,
                title="Satisfacción del Cliente por Aspecto",
                template=template_graficos,
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col4:
            # Gráfico de pie para categorías
            category_sales = products_data.groupby('categoria')['ventas'].sum()
            
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=category_sales.index,
                    values=category_sales.values,
                    hole=0.3,
                    marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99'])
                )
            ])
            
            fig_pie.update_layout(
                title="Ventas por Categoría",
                template=template_graficos,
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabla de datos detallados
        st.markdown("---")
        st.markdown("### Datos Detallados")
        
        detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Ventas Mensuales", "Productos", "Satisfacción"])
        
        with detail_tab1:
            st.dataframe(sales_data, use_container_width=True, hide_index=True)
        
        with detail_tab2:
            # Agregar métricas calculadas
            products_display = products_data.copy()
            products_display['margen_€'] = products_display['ventas'] * products_display['margen']
            products_display['margen'] = products_display['margen'].apply(lambda x: f"{x:.1%}")
            products_display['margen_€'] = products_display['margen_€'].apply(lambda x: f"€{x:,.0f}")
            
            st.dataframe(products_display, use_container_width=True, hide_index=True)
        
        with detail_tab3:
            st.dataframe(satisfaction_data, use_container_width=True, hide_index=True)
        
        # Controles interactivos
        st.markdown("---")
        st.markdown("### Controles Interactivos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Actualizar Datos", key="dashboard_update_data"):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            # Exportar dashboard
            dashboard_data = {
                'ventas_mensuales': sales_data.to_dict('records'),
                'productos': products_data.to_dict('records'),
                'satisfaccion': satisfaction_data.to_dict('records')
            }
            
            st.download_button(
                label="Exportar Dashboard",
                data=json.dumps(dashboard_data, indent=2, default=str),
                file_name="dashboard_data.json",
                mime="application/json",
                key="dashboard_export"
            )
        
        with col3:
            if st.button("Compartir Dashboard", key="dashboard_share"):
                st.success("¡Dashboard compartido! (Función simulada)")
        
        # Código del dashboard
        with st.expander("Ver código del dashboard"):
            st.code('''
# Dashboard completo con múltiples visualizaciones

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Ventas", f"€{total_sales:,}", "12%")

# Gráficos en grid
col1, col2 = st.columns(2)

with col1:
    # Gráfico de líneas
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['fecha'], y=df['ventas']))
    st.plotly_chart(fig)

with col2:
    # Gráfico de barras
    fig = go.Figure(go.Bar(x=df['producto'], y=df['ventas']))
    st.plotly_chart(fig)

# Tabla de datos
st.dataframe(df)

# Controles
if st.button("Actualizar"):
    st.rerun()
            ''', language='python')