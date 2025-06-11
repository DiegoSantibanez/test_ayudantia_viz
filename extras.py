import plotly.graph_objects as go
import numpy as np
import re

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

def show_color_scale(color_scale=None, n_steps=20, save_html=False):
    """
    Muestra visualmente la escala de colores.
    
    Args:
        color_scale: Escala de colores en formato [[posición, 'rgb(r,g,b)'], ...]
                    Si es None, usa COLOR_SCALE por defecto
        n_steps: Número de pasos a mostrar
        save_html: Si guardar el gráfico como HTML
    """
    if color_scale is None:
        color_scale = COLOR_SCALE
    
    x = np.linspace(0, 1, n_steps)
    y = [0] * len(x)
    
    hover_texts = []
    colors_for_plot = []
    
    for pos in x:
        color_rgb = interpolar_color(pos, color_scale)
        hover_texts.append(f'Posición: {pos:.2f}<br>{color_rgb}')
        colors_for_plot.append(color_rgb)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=40,
            color=colors_for_plot,
            showscale=False
        ),
        text=hover_texts,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        template=template_graficos,
        title='Escala de Colores',
        xaxis_title='Posición',
        yaxis_visible=False,
        height=200,
        showlegend=False,
        margin=dict(t=50, b=50)
    )
    
    fig.update_xaxes(
        range=[-0.05, 1.05],
        ticktext=[f'{v:.2f}' for v in np.arange(0, 1.1, 0.1)],
        tickvals=np.arange(0, 1.1, 0.1),
        tickangle=45
    )
    
    fig.update_yaxes(range=[-0.5, 0.5])
    
    if save_html:
        fig.write_html('escala_colores.html')
    
    fig.show()