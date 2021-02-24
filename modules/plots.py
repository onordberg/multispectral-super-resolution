import plotly.express as px
import plotly.graph_objects as go


def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def get_plotly_standard_colors(hex_rgb_rgba, alpha=0.5):
    hex_colors = px.colors.qualitative.Plotly
    if hex_rgb_rgba == 'hex':
        return hex_colors
    colors = []
    for h in hex_colors:
        if hex_rgb_rgba in ('rgb', 'rgba'):
            rgb = hex_to_rgb(h)
            if hex_rgb_rgba == 'rgb':
                colors.append('rgb' + str(rgb))
            elif hex_rgb_rgba == 'rgba':
                colors.append('rgba' + str((rgb[0], rgb[1], rgb[2], alpha)))
    return colors


def rgb_to_rgba(rgb_string, alpha):
    rgba = rgb_string.replace('rgb', 'rgba')
    rgba = rgba.replace(')', ', ' + str(alpha) + ')')
    return rgba
