# -*- coding: utf-8 -*-
# ==========================================================
# REPORTE A4 "BONITO": PÓRTICO 3D paramétrico (OpenSeesPy)
# CARGAS: peso propio (vigas/columnas) + losa + acabados + tabiquería + viva (E.020)
# - Plantas por nivel con columnas rectangulares
# - Vista 3D con prismas (vigas/columnas) y flechas de carga
# - Páginas por elemento: Vigas (Vz, My), Columnas (My, Mz)
# - Diagramas con relleno de líneas VERTICALES
# ==========================================================
# pip install openseespy numpy matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime
import textwrap
import openseespy.opensees as ops
import math

# Constantes de maquetación A4 con márgenes específicos
CM_TO_INCH = 1.0 / 2.54
A4_WIDTH_IN = 8.27
A4_HEIGHT_IN = 11.69

MARGIN_LEFT_CM = 1.0
MARGIN_TOP_CM = 1.0
MARGIN_RIGHT_CM = 1.0
MARGIN_BOTTOM_CM = 1.0

PAGE_LEFT = (MARGIN_LEFT_CM * CM_TO_INCH) / A4_WIDTH_IN
PAGE_RIGHT = 1.0 - (MARGIN_RIGHT_CM * CM_TO_INCH) / A4_WIDTH_IN
PAGE_TOP = 1.0 - (MARGIN_TOP_CM * CM_TO_INCH) / A4_HEIGHT_IN
PAGE_BOTTOM = (MARGIN_BOTTOM_CM * CM_TO_INCH) / A4_HEIGHT_IN

# Posiciones auxiliares para encabezados y pies siempre dentro del marco util
HEADER_OFFSET = 0.018
SUBHEADER_GAP = 0.050
FOOTER_OFFSET = 0.018
PAGE_NUMBER_OFFSET = 0.085

FIGURE_CAPTION_GAP = 0.16
TABLE_CAPTION_GAP = 0.006

PAGE_HEADER_Y = PAGE_TOP - HEADER_OFFSET
PAGE_SUBHEADER_Y = max(PAGE_BOTTOM + FOOTER_OFFSET, PAGE_HEADER_Y - SUBHEADER_GAP)
PAGE_FOOTER_Y = PAGE_BOTTOM + FOOTER_OFFSET


def apply_page_margins(fig):
    fig.subplots_adjust(left=PAGE_LEFT, right=PAGE_RIGHT,
                        top=PAGE_TOP, bottom=PAGE_BOTTOM)


# Registro global de pies de figura y tabla para generar leyendas consistentes
figure_registry = []
table_registry = []
_figure_labels = {}
_table_labels = {}

page_outline_plan = []
page_outline_actual = []
_page_counter = 0


def _reset_caption_registry():
    global figure_registry, table_registry, _figure_labels, _table_labels
    figure_registry = []
    table_registry = []
    _figure_labels = {}
    _table_labels = {}


def reset_page_counter():
    global page_outline_plan, page_outline_actual, _page_counter
    page_outline_plan = []
    page_outline_actual = []
    _page_counter = 0


def register_figure(key, title):
    if key in _figure_labels:
        return _figure_labels[key]['label']
    label = f"Figura N°{len(figure_registry)+1:02d}"
    entry = {'label': label, 'title': title}
    _figure_labels[key] = entry
    figure_registry.append(entry)
    return label


def register_table(key, title):
    if key in _table_labels:
        return _table_labels[key]['label']
    label = f"Tabla N°{len(table_registry)+1:02d}"
    entry = {'label': label, 'title': title}
    _table_labels[key] = entry
    table_registry.append(entry)
    return label


def figure_caption(key):
    entry = _figure_labels.get(key)
    if entry is None:
        raise KeyError(f"Figura con clave {key} no registrada")
    return f"{entry['label']}: {entry['title']}"


def table_caption(key):
    entry = _table_labels.get(key)
    if entry is None:
        raise KeyError(f"Tabla con clave {key} no registrada")
    return f"{entry['label']}: {entry['title']}"


def finalize_page(pdf, fig, page_info, footer_left=None, footer_center=None, footer_above_right=None):
    global _page_counter
    _page_counter += 1
    page_number = _page_counter
    fig.text(PAGE_RIGHT, PAGE_BOTTOM - PAGE_NUMBER_OFFSET,
             f'Página {page_number}', ha='right', va='center', fontsize=9)
    if footer_left:
        fig.text(PAGE_LEFT, PAGE_FOOTER_Y, footer_left,
                 ha='left', va='center', fontsize=9)
    if footer_center:
        fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_FOOTER_Y,
                 footer_center, ha='center', va='center', fontsize=9)
    if footer_above_right:
        fig.text(PAGE_RIGHT, PAGE_FOOTER_Y + 0.03,
                 footer_above_right, ha='right', va='center', fontsize=9)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)
    recorded = dict(page_info)
    recorded['page'] = page_number
    page_outline_actual.append(recorded)

def page_spectrum(modal_x, modal_y):
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    ax = fig.add_subplot(1, 1, 1)

    modal_periods = []
    for modal in (modal_x, modal_y):
        for mode in modal.get('modes', []):
            modal_periods.append(mode['T'])

    Tmax = max([0.5, Tl * 1.4] + modal_periods)
    periods = np.linspace(0.01, Tmax, 800)
    Sa = spectral_accel_e030(periods) / g_grav
    ax.plot(periods, Sa, color='tab:blue', lw=2.5, label='Espectro E.030 (zeta=5%)')

    def plot_modal_points(modal, color, label_prefix):
        modes = modal.get('modes', [])
        if not modes:
            return
        for idx, mode in enumerate(modes[:3], start=1):
            label = f"{label_prefix} modo {idx}: T={mode['T']:.2f}s" if idx == 1 else None
            ax.scatter(mode['T'], mode['Sa'] / g_grav, color=color, s=60, marker='o', label=label)

    plot_modal_points(modal_x, 'tab:orange', 'X')
    plot_modal_points(modal_y, 'tab:green', 'Y')

    ax.set_title('Espectro elastico E.030 (direcciones X e Y)', fontsize=14, weight='bold')
    ax.set_xlabel('Periodo T [s]')
    ax.set_ylabel('Sa [g]')
    ax.grid(True, ls=':', alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)

    resume = [
        f'T1x={Tx1:.3f}s  masa mod={mode_mass_x_pct:.1f}%  Vb={base_shear_x_ton:.2f} t',
        f'T1y={Ty1:.3f}s  masa mod={mode_mass_y_pct:.1f}%  Vb={base_shear_y_ton:.2f} t',
        f'Deriva max: X={drift_x_pct:.2f}%  Y={drift_y_pct:.2f}%'
    ]
    footer = "\n".join(resume)
    return fig, footer



# ------------------ PARÁMETROS EDITABLES ------------------
# Discretización
nx, ny, nz = 2, 2, 2                       # vanos X, vanos Y, pisos
Lx_spans = 5                # len = nx
Ly_spans = 6.0                             # escalar -> se replica ny veces
H_levels = 3.5                     # len = nz

# Predimensionamiento inicial (referencia E.060/E.070)
CATEGORIA_DEFAULT = 'A'
ALPHA_MAP = {'A': 10.0, 'B': 11.0, 'C': 12.0}
Q_USO_TABLA = {'A': 1500.0, 'B': 1300.0, 'C': 1000.0}  # kg/m2
G_ACC = 9.80665
FC_MPA_DEFAULT = 21.0

RHO_CONCRETO = 2400.0                       # kg/m3
GAMMA_CONCRETO = RHO_CONCRETO * G_ACC / 1e3 # kN/m3

# Cargas de piso (kN/m2)
gamma_conc = GAMMA_CONCRETO                 # peso específico hormigón
t_losa     = 0.2                          # espesor de losa (m)
q_losa     = gamma_conc * t_losa            # kN/m2 (peso propio de losa)
q_acab     = 1.00                           # acabados (ajusta)
q_tabiq    = 1.00                           # tabiquería distribuida (ajusta)
q_viva     = 2.00                           # E.020 (vivienda por defecto). Cambia según uso real.

# Por si cada nivel tiene uso distinto: acepta escalar o lista len=nz
qv_levels  = q_viva                         # carga viva por nivel
qd_extra   = q_acab + q_tabiq + q_losa      # DL de piso adicional (sin vigas/columnas)

# ¿Qué niveles reciben las cargas de piso? (p.ej. todos)
loaded_levels = list(range(1, nz+1))        # [1..nz]

# Conversión de unidades base
MPA_TO_KGF_CM2 = 10.197162129779
KGF_CM2_TO_MPA = 1.0 / MPA_TO_KGF_CM2

# Secciónes (para modelo y dibujo)
E_conc_MPa = 25000.0
E_conc_kgf_cm2 = E_conc_MPa * MPA_TO_KGF_CM2
E, nu = E_conc_MPa * 1000.0, 0.20
G      = E/(2*(1+nu))

# Diseño de columnas (materiales y refuerzo)
fc_col_kgf_cm2 = 28.0 * MPA_TO_KGF_CM2    # kg/cm²
fy_col_kgf_cm2 = 420.0 * MPA_TO_KGF_CM2   # kg/cm²
Es_col_kgf_cm2 = 200000.0 * MPA_TO_KGF_CM2  # kg/cm²
fc_col = fc_col_kgf_cm2 * KGF_CM2_TO_MPA
fy_col = fy_col_kgf_cm2 * KGF_CM2_TO_MPA
Es_col = Es_col_kgf_cm2 * KGF_CM2_TO_MPA
clear_cover_col = 0.04 # m (cara de concreto a estribo)
stirrup_diam = 0.010   # m
phi_axial_col = 0.65   # factor resistencia para columna arriostrada
eps_cu = 0.003

# Diseño de vigas (materiales y opciones de refuerzo)
fc_beam_kgf_cm2 = 28.0 * MPA_TO_KGF_CM2   # kg/cm²
fy_beam_kgf_cm2 = 420.0 * MPA_TO_KGF_CM2  # kg/cm²
fc_beam = fc_beam_kgf_cm2 * KGF_CM2_TO_MPA
fy_beam = fy_beam_kgf_cm2 * KGF_CM2_TO_MPA
phi_flex_beam = 0.90   # ACI 318 flexión controlada
clear_cover_beam = 0.04    # m
stirrup_diam_beam = 0.010  # m

beam_bar_sizes = {
    'N5': 0.015875,
    'N6': 0.01905,
    'N8': 0.0254,
}

REBAR_DIAMETER_LABELS = [
    (0.015875, '5/8"'),
    (0.01905, '3/4"'),
    (0.0254, '1"'),
    (0.0381, '1 1/2"'),
]

BEAM_BASE_MIN = 0.30
BEAM_BASE_MAX = 0.70
BEAM_BASE_STEP = 0.05
BEAM_HEIGHT_MIN = 0.40
BEAM_HEIGHT_STEP = 0.05

BEAM_MIN_CLEAR_SPACING = 0.04

BEAM_ALLOWED_DIAMETER_SETS = [
    {beam_bar_sizes['N5']},
    {beam_bar_sizes['N6']},
    {beam_bar_sizes['N8']},
    {beam_bar_sizes['N5'], beam_bar_sizes['N6']},
    {beam_bar_sizes['N6'], beam_bar_sizes['N8']},
]


def bar_diameter_label(diam):
    best = min(REBAR_DIAMETER_LABELS, key=lambda item: abs(item[0] - diam))
    if abs(best[0] - diam) <= 5e-4:
        return best[1]
    return f"{diam * 1000:.0f} mm"

# Soportes en base
base_fix = (1,1,1,1,1,1)

# Salida
PDF_NAME = "Portico3D_Reporte_A4.pdf"

# Sismo (E.030) y combinaciones (E.060)
Z_sismo = 0.45            # Factor de zona sísmica
U_importancia = 1.50      # Factor de uso/importancia
S_suelo = 1.00            # Factor de suelo
R_respuesta = 8.00        # Factor de reducción de respuesta (marcos de concreto)
Tp = 0.40                 # Periodo Tp del espectro (s)
Tl = 2.50                 # Periodo Tl del espectro (s)
C_min = 0.10              # Valor mínimo del coeficiente C
xi_modal = 0.05           # Amortiguamiento modal (5%)
psi_live = 0.25           # Fracción de carga viva que participa en la masa
max_modal_modes = None    # Cambia a un entero para forzar cantidad de modos

# ------------------ UTILIDADES ------------------
def as_list(x, n):
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != n: raise ValueError(f"Se esperaban {n} valores y recibí {len(x)}")
        return list(x)
    return [x]*n


def human_join(items, sep=", ", last=" y "):
    items = [str(item) for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return sep.join(items[:-1]) + last + items[-1]


def ceil_to_step(value, step):
    if step <= 0:
        raise ValueError("El incremento debe ser positivo para redondear.")
    return step * math.ceil((value - 1e-9) / step)


def floor_to_step(value, step):
    if step <= 0:
        raise ValueError("El incremento debe ser positivo para redondear.")
    return step * math.floor((value + 1e-9) / step)


def predimensionar_secciones(nx, ny, nz, Lx, Ly,
                             categoria=CATEGORIA_DEFAULT,
                             fc_mpa=FC_MPA_DEFAULT):
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("La geometría del pórtico debe tener al menos un vano en cada dirección y un nivel.")

    spans_x = [float(val) for val in Lx] if Lx else [0.0]
    spans_y = [float(val) for val in Ly] if Ly else [0.0]

    dx = max(spans_x) if spans_x else 0.0
    dy = max(spans_y) if spans_y else 0.0
    Ln = max(dx, dy, 0.30)

    alpha = ALPHA_MAP.get(categoria.upper(), ALPHA_MAP[CATEGORIA_DEFAULT])
    q_uso = Q_USO_TABLA.get(categoria.upper(), Q_USO_TABLA[CATEGORIA_DEFAULT])
    fc = abs(fc_mpa) * 1e6

    h_raw = Ln / alpha if alpha > 0 else BEAM_HEIGHT_MIN
    h_tmp = max(h_raw, BEAM_HEIGHT_MIN)
    h_beam = ceil_to_step(h_tmp, BEAM_HEIGHT_STEP)

    b_min_required = max(BEAM_BASE_MIN, h_beam / 2.0)
    b_beam = ceil_to_step(b_min_required, BEAM_BASE_STEP)

    if h_beam <= b_beam:
        h_beam = ceil_to_step(max(b_beam + BEAM_HEIGHT_STEP, BEAM_HEIGHT_MIN), BEAM_HEIGHT_STEP)

    if h_beam > 2.0 * b_beam:
        b_beam = ceil_to_step(h_beam / 2.0, BEAM_BASE_STEP)

    if h_beam <= b_beam:
        h_beam = ceil_to_step(max(b_beam + BEAM_HEIGHT_STEP, BEAM_HEIGHT_MIN), BEAM_HEIGHT_STEP)

    trib_area_interior = dx * dy
    trib_area_border = 0.5 * trib_area_interior
    trib_area_corner = 0.25 * trib_area_interior
    worst_area = max(trib_area_interior, trib_area_border, trib_area_corner, 0.01)

    Pserv = q_uso * worst_area * nz * G_ACC
    denom = 0.2 * fc if fc > 0 else 1.0
    Acol = Pserv / denom if denom > 0 else 0.30 ** 2
    lado_raw = max(0.30, math.sqrt(max(Acol, 0.0)))
    lado_col = ceil_to_step(lado_raw, 0.05)

    return {
        'beam': {
            'b': b_beam,
            'h': h_beam,
            'Ln': Ln,
            'alpha': alpha,
            'categoria': categoria,
        },
        'column': {
            'side': lado_col,
            'tributary_area': worst_area,
            'Pserv': Pserv,
            'fc': fc,
        }
    }

Lx = as_list(Lx_spans, nx)
Ly = as_list(Ly_spans, ny)
H  = as_list(H_levels, nz)

predimension = predimensionar_secciones(nx, ny, nz, Lx, Ly)

beam_predim_base = predimension['beam']['b']
beam_predim_height = predimension['beam']['h']

sec_beam_b, sec_beam_h = beam_predim_base, beam_predim_height
Abeam  = sec_beam_b * sec_beam_h
Iybeam = sec_beam_b * sec_beam_h**3 / 12.0
Izbeam = sec_beam_h * sec_beam_b**3 / 12.0
Jbeam  = 1e-3

column_predim_side = predimension['column']['side']
sec_col_b = sec_col_h = column_predim_side
sec_col_plan = (sec_col_b, sec_col_h)
Acol  = sec_col_b * sec_col_h
Iycol = sec_col_b * sec_col_h**3 / 12.0
Izcol = sec_col_h * sec_col_b**3 / 12.0
Jcol  = 1e-3

# Banda en planta para dibujar vigas
beam_plan_width = sec_beam_b
qv = as_list(qv_levels, nz)   # viva por nivel

X = np.cumsum([0.0] + Lx)
Y = np.cumsum([0.0] + Ly)
Z = np.cumsum([0.0] + H)

def node_id(ix, iy, iz):
    return 1 + ix + (nx+1)*iy + (nx+1)*(ny+1)*iz

def level_pair_key(level):
    if level <= 0:
        return (1, min(2, nz))
    start = level if level % 2 == 1 else level - 1
    start = max(1, start)
    end = min(start + 1, nz)
    return (start, end)


def level_pair_label(pair):
    a, b = pair
    if a == b:
        return f"piso {a}"
    return f"pisos {a}-{b}"

def tributary_width_X_row(iy):
    # para viga // X en fila iy → ancho tributario en Y
    dy_left  = (Y[iy]   - Y[iy-1]) if iy > 0   else (Y[1]-Y[0])
    dy_right = (Y[iy+1] - Y[iy])   if iy < ny  else (Y[ny]-Y[ny-1])
    return 0.5*(dy_left + dy_right)

def tributary_width_Y_col(ix):
    # para viga // Y en columna ix → ancho tributario en X
    dx_left  = (X[ix]   - X[ix-1]) if ix > 0   else (X[1]-X[0])
    dx_right = (X[ix+1] - X[ix])   if ix < nx  else (X[nx]-X[nx-1])
    return 0.5*(dx_left + dx_right)

# ------------------ MODELO OPENSEES ------------------
ops.wipe()
ops.model('basic','-ndm',3,'-ndf',6)

# Nodos
for iz in range(nz+1):
    for iy in range(ny+1):
        for ix in range(nx+1):
            ops.node(node_id(ix,iy,iz), X[ix], Y[iy], Z[iz])

# Apoyos base
for iy in range(ny+1):
    for ix in range(nx+1):
        ops.fix(node_id(ix,iy,0), *base_fix)

# Transformaciones
ops.geomTransf('Linear', 1, 0,0,1)  # vigas (z local = Z global)
ops.geomTransf('Linear', 2, 0,1,0)  # columnas

# Elementos
eleTag = 1
columns = []   # (ele, ix, iy, iz)  iz = nivel superior (entre iz-1 → iz)
beamsX  = []   # (ele, iz, iy, ix)
beamsY  = []   # (ele, iz, ix, iy)
beam_info = {}

# Columnas
for iz in range(1, nz+1):
    for iy in range(ny+1):
        for ix in range(nx+1):
            nd_i = node_id(ix,iy,iz-1)
            nd_j = node_id(ix,iy,iz)
            ops.element('elasticBeamColumn', eleTag, nd_i, nd_j,
                        Acol, E, G, Jcol, Iycol, Izcol, 2)
            columns.append((eleTag, ix, iy, iz))
            eleTag += 1

# Vigas X
for iz in range(1, nz+1):
    for iy in range(ny+1):
        for ix in range(nx):
            nd_i = node_id(ix,   iy, iz)
            nd_j = node_id(ix+1, iy, iz)
            ops.element('elasticBeamColumn', eleTag, nd_i, nd_j,
                        Abeam, E, G, Jbeam, Iybeam, Izbeam, 1)
            beamsX.append((eleTag, iz, iy, ix))
            beam_info[eleTag] = {'direction': 'X', 'iz': iz, 'ix': ix, 'iy': iy}
            eleTag += 1

# Vigas Y
for iz in range(1, nz+1):
    for ix in range(nx+1):
        for iy in range(ny):
            nd_i = node_id(ix, iy,   iz)
            nd_j = node_id(ix, iy+1, iz)
            ops.element('elasticBeamColumn', eleTag, nd_i, nd_j,
                        Abeam, E, G, Jbeam, Iybeam, Izbeam, 1)
            beamsY.append((eleTag, iz, ix, iy))
            beam_info[eleTag] = {'direction': 'Y', 'iz': iz, 'ix': ix, 'iy': iy}
            eleTag += 1

columns_by_level = sorted(columns, key=lambda item: (item[3], item[2], item[1], item[0]))

# ------------------ CARGAS BASE, DIAFRAGMA RÍGIDO Y MASAS ------------------
g_grav = G_ACC  # m/s2
kN_TO_TON = 1.0 / 9.80665  # 1 kN = 0.10197 toneladas-fuerza
TON_TO_KN = 1.0 / kN_TO_TON

# Peso propio de vigas (lineal) y mapas de carga por caso
w_self_beam = gamma_conc * Abeam  # kN/m
beam_tags = [ele for (ele, _, _, _) in beamsX] + [ele for (ele, _, _, _) in beamsY]
w_dead_map = {}
w_live_map = {}
for (ele, iz, iy, ix) in beamsX:
    trib = tributary_width_X_row(iy)
    w_dead = w_self_beam
    w_live = 0.0
    if iz in loaded_levels:
        w_dead += qd_extra * trib
        w_live += qv[iz-1] * trib
    w_dead_map[ele] = w_dead
    w_live_map[ele] = w_live
for (ele, iz, ix, iy) in beamsY:
    trib = tributary_width_Y_col(ix)
    w_dead = w_self_beam
    w_live = 0.0
    if iz in loaded_levels:
        w_dead += qd_extra * trib
        w_live += qv[iz-1] * trib
    w_dead_map[ele] = w_dead
    w_live_map[ele] = w_live
w_total_map = {ele: w_dead_map.get(ele, 0.0) + w_live_map.get(ele, 0.0) for ele in beam_tags}
beam_zero_map = {ele: 0.0 for ele in beam_tags}

# Peso propio de columnas como carga nodal (nivel superior)
column_dead_loads = {}
for (ele, ix, iy, iz) in columns_by_level:
    nd_j = node_id(ix, iy, iz)
    Lc = Z[iz] - Z[iz-1]
    w_self_col = gamma_conc * Acol
    P_col = w_self_col * Lc
    column_dead_loads[nd_j] = column_dead_loads.get(nd_j, 0.0) + P_col

# Nodos maestros del diafragma rígido (uno por nivel)
floor_area = X[-1] * Y[-1]
x_cg = 0.5 * X[-1]
y_cg = 0.5 * Y[-1]
floor_master = {}
node_tag_counter = node_id(nx, ny, nz)
for iz in range(1, nz+1):
    node_tag_counter += 1
    master_tag = node_tag_counter
    ops.node(master_tag, x_cg, y_cg, Z[iz])
    ops.fix(master_tag, 0, 0, 1, 1, 1, 0)
    floor_master[iz] = master_tag
    for iy in range(ny+1):
        for ix in range(nx+1):
            slave = node_id(ix, iy, iz)
            ops.equalDOF(master_tag, slave, 1, 2, 6)

# Masas concentradas por piso (kg equivalentes en kN*s^2/m)
floor_mass_data = {iz: {'mass': 0.0, 'Jz': 0.0} for iz in range(1, nz+1)}
radius_sq_plate = (X[-1]**2 + Y[-1]**2) / 12.0

def add_point_mass(level, mass, x, y):
    if level not in floor_mass_data or mass <= 0.0:
        return
    r2 = (x - x_cg)**2 + (y - y_cg)**2
    floor_mass_data[level]['mass'] += mass
    floor_mass_data[level]['Jz'] += mass * r2

def add_uniform_plate_mass(level, mass):
    if level not in floor_mass_data or mass <= 0.0:
        return
    floor_mass_data[level]['mass'] += mass
    floor_mass_data[level]['Jz'] += mass * radius_sq_plate

# Cargas de piso (DL + ψ*LL)
for iz in range(1, nz+1):
    if iz in loaded_levels:
        dead_mass = qd_extra * floor_area / g_grav
        live_mass = psi_live * qv[iz-1] * floor_area / g_grav
        add_uniform_plate_mass(iz, dead_mass + live_mass)

# Masa de vigas (peso propio)
mass_per_length_beam = w_self_beam / g_grav
for (ele, iz, iy, ix) in beamsX:
    L = X[ix+1] - X[ix]
    m = mass_per_length_beam * L
    xc = 0.5 * (X[ix] + X[ix+1])
    yc = Y[iy]
    add_point_mass(iz, m, xc, yc)
for (ele, iz, ix, iy) in beamsY:
    L = Y[iy+1] - Y[iy]
    m = mass_per_length_beam * L
    xc = X[ix]
    yc = 0.5 * (Y[iy] + Y[iy+1])
    add_point_mass(iz, m, xc, yc)

# Masa de columnas (mitad a cada extremo)
mass_per_length_col = gamma_conc * Acol / g_grav
for (ele, ix, iy, iz) in columns_by_level:
    Lc = Z[iz] - Z[iz-1]
    m = mass_per_length_col * Lc
    share = 0.5 * m
    x = X[ix]; y = Y[iy]
    add_point_mass(iz, share, x, y)
    if iz-1 >= 1:
        add_point_mass(iz-1, share, x, y)

floor_mass_summary = {}
total_mass = 0.0
for iz, data in floor_mass_data.items():
    mass = data['mass']
    Jz = data['Jz']
    node_tag = floor_master[iz]
    ops.mass(node_tag, mass, mass, 0.0, 0.0, 0.0, Jz)
    floor_mass_summary[iz] = {'node': node_tag, 'mass': mass, 'Jz': Jz}
    total_mass += mass
total_weight = total_mass * g_grav

beam_lengths = {}
for (ele, iz, iy, ix) in beamsX:
    beam_lengths[ele] = X[ix+1] - X[ix]
for (ele, iz, ix, iy) in beamsY:
    beam_lengths[ele] = Y[iy+1] - Y[iy]

# ------------------ FUNCIONES DE ANÁLISIS ------------------
def spectral_c_e030(T):
    T = np.asarray(T, dtype=float)
    c = np.piecewise(
        T,
        [T < 0.2 * Tp,
         (T >= 0.2 * Tp) & (T < Tp),
         (T >= Tp) & (T < Tl),
         T >= Tl],
        [
            lambda val: 1.0 + 7.5 * (val / Tp),
            2.5,
            lambda val: 2.5 * (Tp / val),
            lambda val: 2.5 * (Tp * Tl / (val**2))
        ]
    )
    return np.maximum(c, C_min)

def spectral_accel_e030(T):
    C = spectral_c_e030(T)
    Sa = (Z_sismo * U_importancia * S_suelo / R_respuesta) * C * g_grav
    return Sa

def configure_linear_static():
    ops.wipeAnalysis()
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.test('NormUnbalance', 1e-9, 20, 0)
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')

def collect_case_results(beam_w_map):
    case_data = {'beams': {}, 'columns': {}, 'displacements': {}}
    for (ele, _, _, _) in beamsX + beamsY:
        f = np.array(ops.eleResponse(ele, 'localForce'), dtype=float)
        Vi = f[2] * kN_TO_TON
        Vj = -f[8] * kN_TO_TON
        Mi = f[4] * kN_TO_TON
        Mj = f[10] * kN_TO_TON
        case_data['beams'][ele] = {
            'Vi': Vi,
            'Vj': Vj,
            'Mi': Mi,
            'Mj': Mj,
            'w': beam_w_map.get(ele, 0.0) * kN_TO_TON
        }
    for (ele, ix, iy, iz) in columns_by_level:
        f = np.array(ops.eleResponse(ele, 'localForce'), dtype=float)
        case_data['columns'][ele] = {
            'Pi': f[0] * kN_TO_TON,
            'Vyi': f[1] * kN_TO_TON,
            'Vzi': f[2] * kN_TO_TON,
            'Pj': -f[6] * kN_TO_TON,
            'Vyj': -f[7] * kN_TO_TON,
            'Vzj': -f[8] * kN_TO_TON,
            'Myi': f[4] * kN_TO_TON,
            'Mzi': f[5] * kN_TO_TON,
            'Myj': f[10] * kN_TO_TON,
            'Mzj': f[11] * kN_TO_TON,
            'nivel': iz
        }
    for iz, node in floor_master.items():
        disp = [ops.nodeDisp(node, dof) for dof in (1, 2, 6)]
        case_data['displacements'][iz] = np.array(disp, dtype=float)
    return case_data

def apply_beam_distributed_loads(load_map):
    for ele, w in load_map.items():
        if abs(w) > 0.0:
            ops.eleLoad('-ele', ele, '-type', '-beamUniform', 0.0, -w)

def apply_column_loads(load_map):
    for node, P in load_map.items():
        if abs(P) > 0.0:
            ops.load(node, 0.0, 0.0, -P, 0.0, 0.0, 0.0)

def apply_seismic_pattern(loads_dict, direction):
    for node, vec in loads_dict.items():
        Fx, Fy, Mz = vec
        if direction == 'X':
            ops.load(node, Fx, 0.0, 0.0, 0.0, 0.0, Mz)
        else:
            ops.load(node, 0.0, Fy, 0.0, 0.0, 0.0, Mz)

def scale_seismic_loads(loads_dict, factor):
    return {node: np.array(vec, dtype=float) * factor for node, vec in loads_dict.items()}

def run_load_case(beam_w_map, column_loads=None, seismic_loads=None, direction=None):
    run_load_case.counter += 1
    ts_tag = run_load_case.counter
    ops.timeSeries('Linear', ts_tag)
    ops.pattern('Plain', ts_tag, ts_tag)
    if beam_w_map:
        apply_beam_distributed_loads(beam_w_map)
    if column_loads:
        apply_column_loads(column_loads)
    if seismic_loads:
        apply_seismic_pattern(seismic_loads, direction)
    configure_linear_static()
    ret = ops.analyze(1)
    if ret != 0:
        ops.test('NormUnbalance', 1e-8, 50, 0)
        ops.algorithm('Newton')
        ret = ops.analyze(1)
        if ret != 0:
            raise RuntimeError("No convergió el análisis estático para el caso de carga.")
    results = collect_case_results(beam_w_map if beam_w_map is not None else beam_zero_map)
    ops.loadConst('-time', 0.0)
    ops.remove('loadPattern', ts_tag)
    ops.remove('timeSeries', ts_tag)
    return results
run_load_case.counter = 100

def hermite_moment_diagram(L, Mi, Mj, Vi, Vj, npts=401):
    if L <= 0.0:
        x = np.zeros(npts)
        return x, np.zeros_like(x)
    x = np.linspace(0.0, L, npts)
    xi = x / L
    h1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    h2 = 3.0 * xi**2 - 2.0 * xi**3
    h3 = xi - 2.0 * xi**2 + xi**3
    h4 = -xi**2 + xi**3
    M = Mi * h1 + Mj * h2 + L * Vi * h3 + L * Vj * h4
    return x, M


def beta1_factor(fc_mpa):
    if fc_mpa <= 28.0:
        return 0.85
    reduction = 0.05 * max(0.0, (fc_mpa - 28.0) / 7.0)
    return max(0.65, 0.85 - reduction)


def build_beam_rebar_options():
    options = []
    max_count = 12
    for size, diam in beam_bar_sizes.items():
        area_bar = 0.25 * np.pi * diam**2
        for count in range(2, max_count + 1):
            label = f"{count}Ø{bar_diameter_label(diam)}"
            options.append({
                'label': label,
                'size': size,
                'count': count,
                'diam': diam,
                'area': count * area_bar
            })
    options.sort(key=lambda item: item['area'])
    return options


beam_rebar_options = build_beam_rebar_options()

beam_rebar_counts = sorted({opt['count'] for opt in beam_rebar_options})
beam_rebar_diams = sorted({opt['diam'] for opt in beam_rebar_options})
beam_rebar_diam_labels = [bar_diameter_label(d) for d in beam_rebar_diams]

beam_rebar_count_text = human_join([f"{count} barras" for count in beam_rebar_counts])
beam_rebar_diam_text = human_join([f"Ø{label}" for label in beam_rebar_diam_labels])

BEAM_STEEL_NOTE = (
    "Configuración replicada de Prueba35.py: catálogo "
    f"{beam_rebar_count_text} por cara con diámetros {beam_rebar_diam_text} "
    "para evaluar refuerzo superior e inferior en todos los tramos."
)

BEAM_STEEL_NOTE_SHORT = (
    "Catálogo Prueba35.py: "
    f"{beam_rebar_counts[0]}–{beam_rebar_counts[-1]} barras {beam_rebar_diam_text} por cara."
)


def _interior_face_positions(count, limit):
    if count <= 0:
        return []
    base = np.linspace(-limit, limit, count + 2)
    return base[1:-1].tolist()


def build_column_rebar_layout(b, h, cover, stirrup, corner_bar_diam, n_bars, face_bar_diam=None):
    face_bar_diam = corner_bar_diam if face_bar_diam is None else face_bar_diam
    max_diam = max(corner_bar_diam, face_bar_diam)
    cover_to_bar = cover + stirrup + max_diam / 2.0
    if cover_to_bar >= min(b, h) / 2.0:
        raise ValueError("La combinacion de recubrimiento y diametros deja sin espacio al acero longitudinal.")
    if n_bars < 8 or (n_bars - 4) % 4 != 0:
        raise ValueError("El numero de barras debe mantener simetria con al menos una barra por cara.")

    bars_per_face = (n_bars - 4) // 4
    y_corner = b / 2.0 - (cover + stirrup + corner_bar_diam / 2.0)
    z_corner = h / 2.0 - (cover + stirrup + corner_bar_diam / 2.0)
    y_face = b / 2.0 - (cover + stirrup + face_bar_diam / 2.0)
    z_face = h / 2.0 - (cover + stirrup + face_bar_diam / 2.0)

    layout = []
    # Barras en las esquinas
    corner_coords = [
        (-y_corner, -z_corner), (-y_corner, z_corner),
        (y_corner, -z_corner), (y_corner, z_corner)
    ]
    for yi, zi in corner_coords:
        area_bar = 0.25 * np.pi * corner_bar_diam**2
        layout.append({'y': yi, 'z': zi, 'area': area_bar, 'diameter': corner_bar_diam})

    # Barras en caras paralelas al eje Y (z constante)
    y_positions = _interior_face_positions(bars_per_face, y_face)
    for z_sign in (-1.0, 1.0):
        zi = z_sign * z_face
        for yi in y_positions:
            area_bar = 0.25 * np.pi * face_bar_diam**2
            layout.append({'y': yi, 'z': zi, 'area': area_bar, 'diameter': face_bar_diam})

    # Barras en caras paralelas al eje X (y constante)
    z_positions = _interior_face_positions(bars_per_face, z_face)
    for y_sign in (-1.0, 1.0):
        yi = y_sign * y_face
        for zi in z_positions:
            area_bar = 0.25 * np.pi * face_bar_diam**2
            layout.append({'y': yi, 'z': zi, 'area': area_bar, 'diameter': face_bar_diam})

    if len(layout) != n_bars:
        raise ValueError("No se pudo construir un arreglo simetrico con la cantidad solicitada de barras.")
    return layout


def classify_column_rebar_layout(layout, tol=1e-6):
    groups = {
        'esquinas': [],
        'caras_y': [],
        'caras_x': [],
        'interiores': []
    }
    if not layout:
        return groups

    max_abs_y = max(abs(bar['y']) for bar in layout)
    max_abs_z = max(abs(bar['z']) for bar in layout)
    # Permite tolerancias relativas para compensar redondeos numéricos
    base_tol = min(max_abs_y, max_abs_z) if max_abs_y > 0 and max_abs_z > 0 else 0.0
    tol_corner = max(tol, 1e-6, 1e-3 * base_tol)

    for bar in layout:
        y_abs = abs(bar['y'])
        z_abs = abs(bar['z'])
        near_max_y = abs(y_abs - max_abs_y) <= tol_corner
        near_max_z = abs(z_abs - max_abs_z) <= tol_corner
        if near_max_y and near_max_z:
            groups['esquinas'].append(bar)
        elif near_max_z:
            groups['caras_y'].append(bar)
        elif near_max_y:
            groups['caras_x'].append(bar)
        else:
            groups['interiores'].append(bar)
    return groups


def summarize_rebar_counts(layout):
    counts = {}
    for bar in layout:
        diam = bar['diameter']
        counts[diam] = counts.get(diam, 0) + 1
    parts = []
    for diam in sorted(counts.keys()):
        label = bar_diameter_label(diam)
        parts.append(f"{counts[diam]}Ø{label}")
    return " + ".join(parts) if parts else ""


def plot_column_section(ax, b, h, cover, stirrup, layout, note=None):
    half_b = b / 2.0
    half_h = h / 2.0
    rect = Rectangle((-half_b, -half_h), b, h, linewidth=1.2, edgecolor='k', facecolor='0.9')
    ax.add_patch(rect)
    stirrup_offset = cover + stirrup / 2.0
    inner = Rectangle((-half_b + stirrup_offset, -half_h + stirrup_offset),
                      b - 2 * stirrup_offset, h - 2 * stirrup_offset,
                      linewidth=1.0, edgecolor='k', facecolor='none', linestyle='--')
    ax.add_patch(inner)
    for bar in layout:
        diam = bar.get('diameter')
        radius = diam / 2.0 if diam is not None else 0.0125
        circ = Circle((bar['y'], bar['z']), radius, color='tab:blue', ec='k', lw=0.6)
        ax.add_patch(circ)

    steel_note = note if note is not None else summarize_rebar_counts(layout)
    ax.text(0.0, half_h + 0.05 * h, steel_note, ha='center', va='bottom',
            fontsize=8.6, color='tab:blue', weight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-half_b * 1.12, half_b * 1.12)
    ax.set_ylim(-half_h * 1.12, half_h * 1.22)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def bar_positions(count, width, cover, stirrup, bar_diam):
    if count <= 1:
        return [0.0]
    free = width - 2 * (cover + stirrup + bar_diam / 2.0)
    free = max(free, 0.0)
    if free <= 0.0:
        spacing = 0.0
    else:
        spacing = free / (count - 1) if count > 1 else 0.0
    start = -free / 2.0
    return [start + i * spacing for i in range(count)]


def plot_beam_section(ax, width, depth, cover, stirrup, bottom_opt, top_opt):
    ax.set_title('Sección transversal', fontsize=10, weight='bold')
    half_b = width / 2.0
    half_h = depth / 2.0
    ax.add_patch(Rectangle((-half_b, -half_h), width, depth, linewidth=1.2, edgecolor='k', facecolor='0.9'))
    stirrup_offset = cover + stirrup / 2.0
    ax.add_patch(Rectangle((-half_b + stirrup_offset, -half_h + stirrup_offset),
                           width - 2 * stirrup_offset, depth - 2 * stirrup_offset,
                           linewidth=1.0, edgecolor='k', facecolor='none', linestyle='--'))

    def place_bars(option, z_coord):
        if option['count'] == 0:
            return
        xs = bar_positions(option['count'], width, cover, stirrup, option['diam'])
        for x in xs:
            ax.add_patch(Circle((x, z_coord), option['diam'] / 2.0, color='tab:blue', ec='k', lw=0.6))

    z_bottom = -half_h + cover + stirrup + (bottom_opt['diam'] / 2.0 if bottom_opt['count'] > 0 else 0.0)
    z_top = half_h - cover - stirrup - (top_opt['diam'] / 2.0 if top_opt['count'] > 0 else 0.0)
    place_bars(bottom_opt, z_bottom)
    place_bars(top_opt, z_top)

    def annotate_face(count, diam, z_coord, valign):
        if count <= 0:
            return
        note = f"{count}Ø{bar_diameter_label(diam)}"
        ax.text(0.0, z_coord + (0.04 if valign == 'bottom' else -0.04) * depth,
                note, ha='center', va=valign, fontsize=9, color='tab:blue', weight='bold')

    annotate_face(bottom_opt['count'], bottom_opt['diam'], z_bottom, 'bottom')
    annotate_face(top_opt['count'], top_opt['diam'], z_top, 'top')

    ax.axhline(0.0, color='0.5', lw=0.6, ls=':')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-half_b * 1.25, half_b * 1.25)
    ax.set_ylim(-half_h * 1.35, half_h * 1.35)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def design_beam_flexure(Mu_ton_m, tension_face, options, b, h, cover, stirrup,
                        fc_mpa, fy_mpa, phi, allowed_diameters=None,
                        min_clear_spacing=BEAM_MIN_CLEAR_SPACING):
    Mu_ton_m = max(float(Mu_ton_m), 0.0)
    Mu_Nm = Mu_ton_m * TON_TO_KN * 1e3
    fc_pa = fc_mpa * 1e6
    fy_pa = fy_mpa * 1e6
    beta1 = beta1_factor(fc_mpa)

    result = {
        'Mu_ton_m': Mu_ton_m,
        'Mu_Nm': Mu_Nm,
        'tension_face': tension_face,
        'option': None,
        'As_req': 0.0,
        'As_min': 0.0,
        'As_max': 0.0,
        'phiMn_Nm': 0.0,
        'phiMn_ton_m': 0.0,
        'ok': False,
        'ratio': 0.0,
        'needs_resize': False
    }

    if Mu_ton_m < 1e-6:
        Mu_Nm = 0.0

    allowed_set = set(allowed_diameters) if allowed_diameters is not None else None

    def effective_depth(diam):
        return h - cover - stirrup - diam / 2.0

    fc_kgf = fc_mpa * MPA_TO_KGF_CM2
    fy_kgf = fy_mpa * MPA_TO_KGF_CM2

    filtered_options = []
    for opt in options:
        if allowed_set is not None and opt['diam'] not in allowed_set:
            continue
        filtered_options.append(opt)

    if filtered_options:
        diam_ref = filtered_options[0]['diam']
    else:
        diam_choices = sorted(allowed_set) if allowed_set else [opt['diam'] for opt in options]
        diam_ref = diam_choices[0] if diam_choices else 0.02

    d_ref = effective_depth(diam_ref)
    b_cm = b * 100.0
    d_cm = d_ref * 100.0
    rho_min = 0.7 * np.sqrt(fc_kgf) / fy_kgf
    As_min_cm2 = rho_min * b_cm * d_cm
    As_min = As_min_cm2 * 1e-4
    rho_b = (0.85 * fc_kgf * beta1 / fy_kgf) * (6000.0 / (6000.0 + fy_kgf))
    As_balanced_cm2 = rho_b * b_cm * d_cm
    As_max = 0.75 * As_balanced_cm2 * 1e-4

    result['As_min'] = As_min
    result['As_max'] = As_max

    if Mu_Nm <= 0.0:
        required_As = As_min
    else:
        d_guess = d_ref
        denom = phi * 0.85 * fc_pa * b * d_guess**2
        demand_ratio = min(1.0, (2.0 * Mu_Nm) / max(denom, 1e-9))
        if demand_ratio >= 1.0 - 1e-6:
            required_As = np.inf
        else:
            As_calc = (0.85 * fc_pa * b * d_guess / fy_pa) * (1.0 - np.sqrt(1.0 - demand_ratio))
            required_As = max(As_calc, As_min)
    result['As_req'] = required_As

    selected = None
    phiMn_selected = 0.0
    feasible_options = []
    for opt in filtered_options:
        count = opt['count']
        diam = opt['diam']
        free_width = b - 2.0 * (cover + stirrup + diam / 2.0)
        if free_width < -1e-9:
            continue
        if count > 1:
            if free_width <= 0.0:
                continue
            spacing = free_width / (count - 1)
            clear_spacing = spacing - diam
            if clear_spacing < min_clear_spacing - 1e-9:
                continue
        feasible_options.append(opt)

    if not feasible_options:
        result['needs_resize'] = True
        return result

    for opt in feasible_options:
        d_eff = effective_depth(opt['diam'])
        if d_eff <= 0.0:
            continue
        a = opt['area'] * fy_pa / (0.85 * fc_pa * b)
        Mn_Nm = opt['area'] * fy_pa * max(d_eff - a / 2.0, 0.0)
        phiMn = phi * Mn_Nm
        if required_As <= opt['area'] + 1e-10 and phiMn >= Mu_Nm - 1e-6:
            selected = opt
            phiMn_selected = phiMn
            break
    if selected is None and feasible_options:
        selected = feasible_options[-1]
        d_eff = effective_depth(selected['diam'])
        a = selected['area'] * fy_pa / (0.85 * fc_pa * b)
        Mn_Nm = selected['area'] * fy_pa * max(d_eff - a / 2.0, 0.0)
        phiMn_selected = phi * Mn_Nm
        if phiMn_selected < Mu_Nm - 1e-6:
            result['needs_resize'] = True

    if selected is None:
        return result

    result['option'] = selected
    result['phiMn_Nm'] = phiMn_selected
    result['phiMn_ton_m'] = phiMn_selected / (TON_TO_KN * 1e3)
    result['As_prov'] = selected['area']
    if Mu_Nm > 0.0:
        result['ratio'] = Mu_Nm / max(phiMn_selected, 1e-6)
    else:
        result['ratio'] = required_As / max(selected['area'], 1e-9)
    result['ok'] = phiMn_selected >= Mu_Nm - 1e-6 and (selected['area'] >= As_min - 1e-9)
    if selected['area'] > As_max + 1e-9:
        result['needs_resize'] = True
    return result


def design_beam_rebar_pair(Mu_pos, Mu_neg, options, b, h, cover, stirrup,
                           fc_mpa, fy_mpa, phi):
    best = None
    best_metric = None
    for diam_set in BEAM_ALLOWED_DIAMETER_SETS:
        bottom = design_beam_flexure(Mu_pos, 'bottom', options, b, h, cover, stirrup,
                                     fc_mpa, fy_mpa, phi, allowed_diameters=diam_set)
        top = design_beam_flexure(Mu_neg, 'top', options, b, h, cover, stirrup,
                                  fc_mpa, fy_mpa, phi, allowed_diameters=diam_set)
        if bottom.get('option') is None or top.get('option') is None:
            continue
        diameters_used = set()
        if bottom['option']['count'] > 0:
            diameters_used.add(bottom['option']['diam'])
        if top['option']['count'] > 0:
            diameters_used.add(top['option']['diam'])
        if len(diameters_used) > 2:
            continue
        if not diameters_used.issubset(diam_set):
            continue
        ok = (bottom['ok'] and top['ok'] and
              not bottom['needs_resize'] and not top['needs_resize'])
        max_ratio = max(bottom.get('ratio', 0.0), top.get('ratio', 0.0))
        total_area = bottom.get('As_prov', 0.0) + top.get('As_prov', 0.0)
        diam_count = len(diameters_used)
        max_diam = max(diameters_used) if diameters_used else 0.0
        metric = (
            0 if ok else 1,
            abs(1.0 - max_ratio),
            total_area,
            diam_count,
            -max_diam,
        )
        if best is None or metric < best_metric:
            best = {
                'bottom': bottom,
                'top': top,
                'ok': ok,
                'diameters': diameters_used,
                'allowed_set': diam_set,
                'max_ratio': max_ratio,
                'total_area': total_area,
            }
            best_metric = metric
    return best


def select_beam_section_for_group(Mu_pos, Mu_neg):
    base0 = ceil_to_step(max(beam_predim_base, BEAM_BASE_MIN), BEAM_BASE_STEP)
    height0 = ceil_to_step(max(beam_predim_height, BEAM_HEIGHT_MIN), BEAM_HEIGHT_STEP)

    min_height0 = ceil_to_step(max(base0 + BEAM_HEIGHT_STEP, BEAM_HEIGHT_MIN), BEAM_HEIGHT_STEP)
    if height0 < min_height0:
        height0 = min_height0
    max_height0 = ceil_to_step(2.0 * base0, BEAM_HEIGHT_STEP)
    if height0 > max_height0:
        height0 = max_height0
    if height0 <= base0:
        height0 = ceil_to_step(base0 + BEAM_HEIGHT_STEP, BEAM_HEIGHT_STEP)

    max_base_steps = max(0, int(math.floor((BEAM_BASE_MAX - base0 + 1e-9) / BEAM_BASE_STEP)))
    max_iterations = max(1, 2 * max_base_steps + 12)

    attempts = []
    seen_geometries = set()
    best_failed = None
    best_failed_metric = None

    def geometry_from_steps(base_steps, height_steps):
        b_raw = base0 + base_steps * BEAM_BASE_STEP
        if b_raw > BEAM_BASE_MAX + 1e-9:
            return None
        b = ceil_to_step(b_raw, BEAM_BASE_STEP)
        b = min(b, BEAM_BASE_MAX)

        h_target = height0 + height_steps * BEAM_HEIGHT_STEP
        h = ceil_to_step(max(h_target, BEAM_HEIGHT_MIN), BEAM_HEIGHT_STEP)
        min_h = ceil_to_step(max(b + BEAM_HEIGHT_STEP, BEAM_HEIGHT_MIN), BEAM_HEIGHT_STEP)
        if h < min_h:
            h = min_h
        max_h = ceil_to_step(2.0 * b, BEAM_HEIGHT_STEP)
        if h > max_h:
            h = max_h
        if h > 2.0 * b + 1e-9:
            h = 2.0 * b
        if h <= b + 1e-9:
            return None
        h = round(h, 3)
        b = round(b, 3)
        if abs(h - b) < 1e-9:
            return None
        return b, h

    for idx in range(max_iterations):
        base_steps = (idx + 1) // 2
        if base_steps > max_base_steps:
            base_steps = max_base_steps
        height_steps = idx // 2
        geom = geometry_from_steps(base_steps, height_steps)
        if geom is None:
            continue
        if geom in seen_geometries:
            continue
        seen_geometries.add(geom)
        b, h = geom
        combo = design_beam_rebar_pair(Mu_pos, Mu_neg, beam_rebar_options,
                                       b, h,
                                       clear_cover_beam, stirrup_diam_beam,
                                       fc_beam, fy_beam, phi_flex_beam)
        if combo is None:
            ratio = math.inf
            ok = False
        else:
            ratio = combo.get('max_ratio', math.inf)
            ok = combo.get('ok', False)
        attempts.append({
            'b': b,
            'h': h,
            'base_steps': base_steps,
            'height_steps': height_steps,
            'ok': ok,
            'max_ratio': None if not math.isfinite(ratio) else ratio
        })
        if ok:
            return {
                'b': b,
                'h': h,
                'ok': True,
                'max_ratio': combo.get('max_ratio', None) if combo else None,
                'combo': combo,
                'attempts': attempts
            }
        metric_ratio = ratio if math.isfinite(ratio) else float('inf')
        metric_area = b * h
        metric = (metric_ratio, metric_area)
        if best_failed is None or metric < best_failed_metric:
            best_failed = {
                'b': b,
                'h': h,
                'combo': combo,
                'max_ratio': combo.get('max_ratio', None) if combo else None
            }
            best_failed_metric = metric

    if best_failed is None:
        best_failed = {
            'b': round(base0, 3),
            'h': round(height0, 3),
            'combo': None,
            'max_ratio': None
        }
    result = {
        'b': best_failed['b'],
        'h': best_failed['h'],
        'ok': False,
        'max_ratio': best_failed.get('max_ratio'),
        'combo': best_failed.get('combo'),
        'attempts': attempts
    }
    return result


Pn0_limit_ton = None


def compute_modal_forces(direction_label, dir_index):
    master_nodes = [floor_master[iz] for iz in sorted(floor_master)]
    if not master_nodes:
        return {'forces': {}, 'modes': [], 'base_shear': 0.0, 'total_mass_dir': 0.0}
    ops.wipeAnalysis()
    ops.system('FullGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    node_masses = {node: np.array(ops.nodeMass(node), dtype=float) for node in master_nodes}
    total_mass_dir = sum(node_masses[node][dir_index] for node in master_nodes)
    target_modes = len(master_nodes) * 3
    if max_modal_modes is not None:
        target_modes = min(target_modes, max_modal_modes)
    eigen_vals = ops.eigen('-fullGenLapack', target_modes)
    modes = []
    for mode_idx, lam in enumerate(eigen_vals, start=1):
        if lam <= 0.0:
            continue
        omega = lam**0.5
        T = 2.0 * np.pi / omega
        phi = {}
        for node in master_nodes:
            phi[node] = np.array([ops.nodeEigenvector(node, mode_idx, dof) for dof in (1, 2, 6)], dtype=float)
        num = sum(node_masses[node][dir_index] * phi[node][dir_index] for node in master_nodes)
        den = sum(node_masses[node][dir_index] * phi[node][dir_index]**2 for node in master_nodes)
        if den <= 0.0 or abs(num) < 1e-12:
            continue
        Gamma = num / den
        Sa = spectral_accel_e030(T)
        forces = {}
        for node in master_nodes:
            masses = node_masses[node]
            phi_vec = phi[node]
            Fx = masses[0] * phi_vec[0] * Gamma * Sa
            Fy = masses[1] * phi_vec[1] * Gamma * Sa
            Mz = masses[5] * phi_vec[2] * Gamma * Sa
            forces[node] = np.array([Fx, Fy, Mz], dtype=float)
        modal_mass_eff = Gamma * num
        mass_ratio = modal_mass_eff / total_mass_dir if total_mass_dir > 0 else 0.0
        modes.append({
            'id': mode_idx,
            'T': T,
            'omega': omega,
            'Gamma': Gamma,
            'Sa': Sa,
            'forces': forces,
            'mass_ratio': mass_ratio
        })
    if not modes:
        return {'forces': {}, 'modes': [], 'base_shear': 0.0, 'total_mass_dir': total_mass_dir}
    dominant_idx = max(range(len(modes)), key=lambda i: modes[i]['mass_ratio'])
    combined = {node: np.zeros(3) for node in master_nodes}
    for comp in range(3):
        for node in master_nodes:
            sum_sq = sum(m['forces'][node][comp]**2 for m in modes)
            sign = np.sign(modes[dominant_idx]['forces'][node][comp])
            if sign == 0.0:
                sign = 1.0
            combined[node][comp] = sign * np.sqrt(sum_sq)
    for idx, mode in enumerate(modes):
        mode['cum_mass_ratio'] = sum(m['mass_ratio'] for m in modes[:idx+1])
    base_shear = sum(combined[node][dir_index] for node in master_nodes)
    return {
        'forces': combined,
        'modes': modes,
        'base_shear': base_shear,
        'total_mass_dir': total_mass_dir,
        'direction': direction_label
    }

# ------------------ CASOS DE CARGA BÁSICOS ------------------
modal_X = compute_modal_forces('X', 0)
modal_Y = compute_modal_forces('Y', 1)

case_results = {}
case_results['D'] = run_load_case(w_dead_map, column_loads=column_dead_loads)
case_results['L'] = run_load_case(w_live_map)
case_results['E_X_POS'] = run_load_case(beam_zero_map, seismic_loads=modal_X['forces'], direction='X')
case_results['E_X_NEG'] = run_load_case(beam_zero_map, seismic_loads=scale_seismic_loads(modal_X['forces'], -1.0), direction='X')
case_results['E_Y_POS'] = run_load_case(beam_zero_map, seismic_loads=modal_Y['forces'], direction='Y')
case_results['E_Y_NEG'] = run_load_case(beam_zero_map, seismic_loads=scale_seismic_loads(modal_Y['forces'], -1.0), direction='Y')

# ------------------ COMBINACIONES E.060 Y ENVOLVENTES ------------------
load_combinations = {
    '1.4CM+1.7CV': {'D': 1.40, 'L': 1.70},
    '1.25(CM+CV)+CSx': {'D': 1.25, 'L': 1.25, 'E_X_POS': 1.00},
    '1.25(CM+CV)-CSx': {'D': 1.25, 'L': 1.25, 'E_X_NEG': 1.00},
    '1.25(CM+CV)+CSy': {'D': 1.25, 'L': 1.25, 'E_Y_POS': 1.00},
    '1.25(CM+CV)-CSy': {'D': 1.25, 'L': 1.25, 'E_Y_NEG': 1.00},
    '0.9CM+CSx': {'D': 0.90, 'E_X_POS': 1.00},
    '0.9CM-CSx': {'D': 0.90, 'E_X_NEG': 1.00},
    '0.9CM+CSy': {'D': 0.90, 'E_Y_POS': 1.00},
    '0.9CM-CSy': {'D': 0.90, 'E_Y_NEG': 1.00},
}

beam_envelopes = {}
for ele in beam_tags:
    L = beam_lengths[ele]
    x = np.linspace(0.0, L, 401)
    V_pos = np.full_like(x, -np.inf)
    V_neg = np.full_like(x, np.inf)
    M_pos = np.full_like(x, -np.inf)
    M_neg = np.full_like(x, np.inf)
    for combo, factors in load_combinations.items():
        Vi = Vj = Mi = Mj = 0.0
        for case, coef in factors.items():
            data = case_results[case]
            blk = data['beams'][ele]
            Vi += coef * blk['Vi']
            Vj += coef * blk['Vj']
            Mi += coef * blk['Mi']
            Mj += coef * blk['Mj']
        x_local, M_curve = hermite_moment_diagram(L, Mi, Mj, Vi, Vj)
        V_curve = np.gradient(M_curve, x_local, edge_order=2) if L > 0 else np.zeros_like(M_curve)
        V_pos = np.maximum(V_pos, V_curve)
        V_neg = np.minimum(V_neg, V_curve)
        M_pos = np.maximum(M_pos, M_curve)
        M_neg = np.minimum(M_neg, M_curve)
    beam_envelopes[ele] = {
        'x': x,
        'V_pos': V_pos,
        'V_neg': V_neg,
        'M_pos': M_pos,
        'M_neg': M_neg
    }

beam_mu_demands = {}
for ele in beam_tags:
    env = beam_envelopes[ele]
    Mu_pos = float(np.max(env['M_pos'])) if env['M_pos'].size else 0.0
    Mu_neg = float(-np.min(env['M_neg'])) if env['M_neg'].size else 0.0
    beam_mu_demands[ele] = {'Mu_pos': Mu_pos, 'Mu_neg': Mu_neg}

beam_group_demands = {}
for ele, mu_data in beam_mu_demands.items():
    info = beam_info.get(ele, {})
    iz = info.get('iz', 1)
    pair_key = level_pair_key(iz)
    stats = beam_group_demands.setdefault(pair_key, {'Mu_pos': 0.0, 'Mu_neg': 0.0})
    stats['Mu_pos'] = max(stats['Mu_pos'], mu_data['Mu_pos'])
    stats['Mu_neg'] = max(stats['Mu_neg'], mu_data['Mu_neg'])

beam_section_pairs = {}
for pair_key, stats in beam_group_demands.items():
    selection = select_beam_section_for_group(stats['Mu_pos'], stats['Mu_neg'])
    beam_section_pairs[pair_key] = {
        'b': selection['b'],
        'h': selection['h'],
        'ok': selection['ok'],
        'Mu_pos': stats['Mu_pos'],
        'Mu_neg': stats['Mu_neg'],
        'max_ratio': selection.get('max_ratio'),
        'attempts': selection.get('attempts', [])
    }

beam_section_by_level = {}
for level in range(1, nz+1):
    pair_key = level_pair_key(level)
    section = beam_section_pairs.get(pair_key)
    if section is None:
        section = {'b': sec_beam_b, 'h': sec_beam_h, 'ok': False, 'Mu_pos': 0.0, 'Mu_neg': 0.0}
    beam_section_by_level[level] = section

beam_section_by_element = {}
beam_design_results = {}
for ele in beam_tags:
    info = beam_info.get(ele, {})
    iz = info.get('iz', 1)
    pair_key = level_pair_key(iz)
    section = beam_section_by_level.get(iz, beam_section_pairs.get(pair_key))
    if section is None:
        section = {'b': sec_beam_b, 'h': sec_beam_h, 'ok': False, 'Mu_pos': 0.0, 'Mu_neg': 0.0}
    b = section['b']
    h = section['h']
    Mu_pos = beam_mu_demands[ele]['Mu_pos']
    Mu_neg = beam_mu_demands[ele]['Mu_neg']
    combo = design_beam_rebar_pair(Mu_pos, Mu_neg, beam_rebar_options,
                                   b, h,
                                   clear_cover_beam, stirrup_diam_beam,
                                   fc_beam, fy_beam, phi_flex_beam)
    if combo is None:
        bottom_design = design_beam_flexure(Mu_pos, 'bottom', beam_rebar_options,
                                            b, h,
                                            clear_cover_beam, stirrup_diam_beam,
                                            fc_beam, fy_beam, phi_flex_beam)
        top_design = design_beam_flexure(Mu_neg, 'top', beam_rebar_options,
                                         b, h,
                                         clear_cover_beam, stirrup_diam_beam,
                                         fc_beam, fy_beam, phi_flex_beam)
        combo_ok = False
        used_diams = set()
    else:
        bottom_design = combo['bottom']
        top_design = combo['top']
        combo_ok = combo['ok']
        used_diams = combo['diameters']
    section_ok = section.get('ok', False)
    ok = bottom_design['ok'] and top_design['ok'] and section_ok and combo_ok
    needs_resize = (bottom_design['needs_resize'] or top_design['needs_resize'] or
                    not section_ok or not combo_ok)
    pair_label = level_pair_label(pair_key)
    beam_design_results[ele] = {
        'Mu_pos': Mu_pos,
        'Mu_neg': Mu_neg,
        'bottom': bottom_design,
        'top': top_design,
        'ok': ok,
        'needs_resize': needs_resize,
        'b': b,
        'h': h,
        'pair_key': pair_key,
        'pair_label': pair_label,
        'section_ok': section_ok,
        'section_ratio': section.get('max_ratio'),
        'diameters': sorted(used_diams)
    }
    beam_section_by_element[ele] = {'b': b, 'h': h, 'pair': pair_key}

beam_section_summary_entries = []
for pair_key in sorted(beam_section_pairs.keys()):
    info = beam_section_pairs[pair_key]
    label = level_pair_label(pair_key)
    status = 'OK' if info['ok'] else 'No cumple'
    ratio = info.get('max_ratio')
    if ratio is not None:
        status = f"{status}, Mu/φMn={ratio:.2f}"
    beam_section_summary_entries.append(
        f"{label}: b={info['b']:.2f} m × h={info['h']:.2f} m ({status})"
    )
beam_section_summary_text = '; '.join(beam_section_summary_entries) if beam_section_summary_entries else '—'

beam_summary_rows = []
beam_order_entries = []
for (ele, iz, iy, ix) in beamsX:
    design = beam_design_results[ele]
    bottom_label = design['bottom']['option']['label'] if design['bottom']['option'] else '—'
    top_label = design['top']['option']['label'] if design['top']['option'] else '—'
    max_ratio = max(design['bottom']['ratio'], design['top']['ratio'])
    section_txt = f"{design['b']:.2f}×{design['h']:.2f}"
    beam_summary_rows.append([
        ele,
        'X',
        f"nivel {iz}, y={iy}, vano x={ix}",
        design['pair_label'],
        section_txt,
        f"{design['Mu_pos']:.2f}",
        f"{design['Mu_neg']:.2f}",
        bottom_label,
        top_label,
        f"{max_ratio:.2f}",
        'OK' if design['ok'] else 'No cumple'
    ])
    beam_order_entries.append((
        max_ratio,
        ele,
        f"Viga-X (nivel {iz}, y={iy}, vano x={ix}, ele={ele}) · {design['pair_label']} · b={design['b']:.2f}×h={design['h']:.2f} m"
    ))

for (ele, iz, ix, iy) in beamsY:
    design = beam_design_results[ele]
    bottom_label = design['bottom']['option']['label'] if design['bottom']['option'] else '—'
    top_label = design['top']['option']['label'] if design['top']['option'] else '—'
    max_ratio = max(design['bottom']['ratio'], design['top']['ratio'])
    section_txt = f"{design['b']:.2f}×{design['h']:.2f}"
    beam_summary_rows.append([
        ele,
        'Y',
        f"nivel {iz}, x={ix}, vano y={iy}",
        design['pair_label'],
        section_txt,
        f"{design['Mu_pos']:.2f}",
        f"{design['Mu_neg']:.2f}",
        bottom_label,
        top_label,
        f"{max_ratio:.2f}",
        'OK' if design['ok'] else 'No cumple'
    ])
    beam_order_entries.append((
        max_ratio,
        ele,
        f"Viga-Y (nivel {iz}, x={ix}, vano y={iy}, ele={ele}) · {design['pair_label']} · b={design['b']:.2f}×h={design['h']:.2f} m"
    ))

ordered_beams = sorted(beam_order_entries, reverse=True)

column_envelopes = {}
for (ele, ix, iy, iz) in columns_by_level:
    L = Z[iz] - Z[iz-1]
    z = np.linspace(0.0, L, 401)
    My_pos = np.full_like(z, -np.inf)
    My_neg = np.full_like(z, np.inf)
    Mz_pos = np.full_like(z, -np.inf)
    Mz_neg = np.full_like(z, np.inf)
    Vy_pos = np.full_like(z, -np.inf)
    Vy_neg = np.full_like(z, np.inf)
    Vz_pos = np.full_like(z, -np.inf)
    Vz_neg = np.full_like(z, np.inf)
    for combo, factors in load_combinations.items():
        Myi = Myj = Mzi = Mzj = 0.0
        Vyi = Vyj = Vzi = Vzj = 0.0
        for case, coef in factors.items():
            blk = case_results[case]['columns'][ele]
            Myi += coef * blk['Myi']
            Myj += coef * blk['Myj']
            Mzi += coef * blk['Mzi']
            Mzj += coef * blk['Mzj']
            Vyi += coef * blk['Vyi']
            Vyj += coef * blk['Vyj']
            Vzi += coef * blk['Vzi']
            Vzj += coef * blk['Vzj']
        if L > 0:
            My_curve = Myi + (Myj - Myi) * (z / L)
            Mz_curve = Mzi + (Mzj - Mzi) * (z / L)
            Vy_curve = Vyi + (Vyj - Vyi) * (z / L)
            Vz_curve = Vzi + (Vzj - Vzi) * (z / L)
        else:
            My_curve = np.full_like(z, Myi)
            Mz_curve = np.full_like(z, Mzi)
            Vy_curve = np.full_like(z, Vyi)
            Vz_curve = np.full_like(z, Vzi)
        My_pos = np.maximum(My_pos, My_curve)
        My_neg = np.minimum(My_neg, My_curve)
        Mz_pos = np.maximum(Mz_pos, Mz_curve)
        Mz_neg = np.minimum(Mz_neg, Mz_curve)
        Vy_pos = np.maximum(Vy_pos, Vy_curve)
        Vy_neg = np.minimum(Vy_neg, Vy_curve)
        Vz_pos = np.maximum(Vz_pos, Vz_curve)
        Vz_neg = np.minimum(Vz_neg, Vz_curve)
    column_envelopes[ele] = {
        'z': z,
        'My_pos': My_pos,
        'My_neg': My_neg,
        'Mz_pos': Mz_pos,
        'Mz_neg': Mz_neg,
        'Vy_pos': Vy_pos,
        'Vy_neg': Vy_neg,
        'Vz_pos': Vz_pos,
        'Vz_neg': Vz_neg,
        'nivel': iz
    }

combo_names = list(load_combinations.keys())
column_combo_forces = {}
for (ele, ix, iy, iz) in columns_by_level:
    column_combo_forces[ele] = {
        'i': {},
        'j': {},
        'meta': {
            'ix': ix,
            'iy': iy,
            'iz': iz,
            'z_bottom': Z[iz-1],
            'z_top': Z[iz]
        }
    }
    for combo, factors in load_combinations.items():
        Pi = Pj = Myi = Myj = Mzi = Mzj = 0.0
        Vyi = Vyj = Vzi = Vzj = 0.0
        for case, coef in factors.items():
            blk = case_results[case]['columns'][ele]
            Pi += coef * blk['Pi']
            Pj += coef * blk['Pj']
            Myi += coef * blk['Myi']
            Myj += coef * blk['Myj']
            Mzi += coef * blk['Mzi']
            Mzj += coef * blk['Mzj']
            Vyi += coef * blk['Vyi']
            Vyj += coef * blk['Vyj']
            Vzi += coef * blk['Vzi']
            Vzj += coef * blk['Vzj']
        Pu = 0.5 * (Pi + Pj)
        column_combo_forces[ele]['i'][combo] = {
            'Pu': Pu,
            'P_end': Pi,
            'Vy': Vyi,
            'Vz': Vzi,
            'My': Myi,
            'Mz': Mzi
        }
        column_combo_forces[ele]['j'][combo] = {
            'Pu': Pu,
            'P_end': Pj,
            'Vy': Vyj,
            'Vz': Vzj,
            'My': Myj,
            'Mz': Mzj
        }

def section_response_uniaxial(axis, c, rebar_layout, b, h, compression_limit_ton,
                              fc=fc_col, fy=fy_col, Es=Es_col, eps_c=eps_cu):
    depth = h if axis == 'y' else b
    width = b if axis == 'y' else h
    coord_key = 'z' if axis == 'y' else 'y'
    extreme = depth / 2.0
    beta1 = beta1_factor(fc)
    c_eff = max(c, 1e-6)
    a = beta1 * c_eff
    if a <= 0.0:
        Cc_kN = 0.0
        uc = extreme
    elif a >= depth:
        Cc_kN = 0.85 * fc * width * depth * 1000.0
        uc = 0.0
    else:
        Cc_kN = 0.85 * fc * width * a * 1000.0
        uc = extreme - a / 2.0
    Pn_kN = Cc_kN
    Mn_kN_m = Cc_kN * uc
    eps_tension = 0.0
    eps_y = fy / Es
    for bar in rebar_layout:
        coord = bar[coord_key]
        dist = extreme - coord
        strain = eps_c * (1.0 - dist / c_eff)
        stress = np.clip(Es * strain, -fy, fy)
        Fs_kN = stress * bar['area'] * 1000.0
        Pn_kN += Fs_kN
        Mn_kN_m += Fs_kN * coord
        eps_tension = min(eps_tension, strain)
    eps_t = abs(min(eps_tension, 0.0))
    if eps_t <= eps_y:
        phi_m = 0.65
    elif eps_t >= eps_y + 0.003:
        phi_m = 0.90
    else:
        phi_m = 0.65 + (eps_t - eps_y) * (0.25 / 0.003)
    phi_m = np.clip(phi_m, 0.65, 0.90)
    Pn_ton = Pn_kN * kN_TO_TON
    Mn_ton_m = Mn_kN_m * kN_TO_TON
    phiMn_ton_m = phi_m * abs(Mn_ton_m)
    compression_limit = compression_limit_ton if compression_limit_ton is not None else np.inf
    if Pn_ton >= 0.0:
        phiPn_ton = phi_m * min(Pn_ton, compression_limit)
    else:
        phiPn_ton = phi_m * Pn_ton
    return {
        'Pn': Pn_ton,
        'Mn': abs(Mn_ton_m),
        'phiMn': phiMn_ton_m,
        'phiPn': phiPn_ton,
        'phi_m': phi_m,
        'eps_t': eps_t
    }

def compute_uniaxial_curve(axis, rebar_layout, b, h, compression_limit_ton, npts=240):
    depth = h if axis == 'y' else b
    c_values = np.linspace(0.01, depth * 6.0, npts)
    data = {'Pn': [], 'phiPn': [], 'Mn': [], 'phiMn': [], 'phi_m': [], 'eps_t': []}
    for c in c_values:
        res = section_response_uniaxial(axis, c, rebar_layout, b, h, compression_limit_ton)
        for key in data:
            data[key].append(res[key])
    steel_area = sum(bar['area'] for bar in rebar_layout)
    tension_capacity_ton = -fy_col * steel_area * 1000.0 * kN_TO_TON
    phi_tension = 0.90
    data['Pn'].append(tension_capacity_ton)
    data['phiPn'].append(phi_tension * tension_capacity_ton)
    data['Mn'].append(0.0)
    data['phiMn'].append(0.0)
    data['phi_m'].append(phi_tension)
    data['eps_t'].append(fy_col / Es_col + 0.003)
    return {key: np.array(values, dtype=float) for key, values in data.items()}

class MomentCapacityCurve:
    def __init__(self, P_values, phiM_values, phiPn0):
        raw_P = np.array(P_values, dtype=float)
        raw_M = np.maximum(np.array(phiM_values, dtype=float), 0.0)

        self.sample_P = np.append(raw_P, phiPn0)
        self.sample_M = np.append(raw_M, 0.0)

        P_aug = self.sample_P
        M_aug = np.maximum(self.sample_M, 0.0)
        order = np.argsort(P_aug)
        P_sorted = P_aug[order]
        M_sorted = M_aug[order]

        grouped = []
        current_p = None
        current_ms = []
        for p, m in zip(P_sorted, M_sorted):
            if current_p is None or abs(p - current_p) >= 1e-4:
                if current_p is not None:
                    grouped.append((current_p, current_ms))
                current_p = float(p)
                current_ms = [float(m)]
            else:
                current_ms.append(float(m))
        if current_p is not None:
            grouped.append((current_p, current_ms))

        unique_P = []
        unique_M = []
        poly_points = []
        for p, ms in grouped:
            ms_sorted = sorted(set(ms))
            if not ms_sorted:
                continue
            unique_P.append(p)
            unique_M.append(ms_sorted[-1])
            for m in ms_sorted:
                poly_points.append((p, m))

        self.P = np.array(unique_P, dtype=float)
        self.M = np.array(unique_M, dtype=float)
        self.poly_points = poly_points
        self.P_min = float(self.P[0])
        self.P_max = float(self.P[-1])

        if self.sample_M.size:
            plateau_mask = np.isclose(self.sample_P, self.P_max, atol=1e-6)
            if np.any(plateau_mask):
                plateau_M = np.maximum(self.sample_M[plateau_mask], 0.0)
                if plateau_M.size:
                    self.plateau_P = float(self.P_max)
                    self.plateau_Mmin = float(np.min(plateau_M))
                    self.plateau_Mmax = float(np.max(plateau_M))
                else:
                    self.plateau_P = None
                    self.plateau_Mmin = None
                    self.plateau_Mmax = None
            else:
                self.plateau_P = None
                self.plateau_Mmin = None
                self.plateau_Mmax = None
        else:
            self.plateau_P = None
            self.plateau_Mmin = None
            self.plateau_Mmax = None

    def phiMn(self, Pu):
        if Pu <= self.P[0]:
            return self.M[0]
        if Pu >= self.P[-1]:
            return self.M[-1]
        return float(np.interp(Pu, self.P, self.M))

    def plateau_segment(self):
        if (self.plateau_P is None or self.plateau_Mmin is None
                or self.plateau_Mmax is None):
            return None
        return {
            'Pu': self.plateau_P,
            'M_min': self.plateau_Mmin,
            'M_max': self.plateau_Mmax
        }

def bresler_alpha(Pu, phiPn0):
    if phiPn0 <= 1e-6:
        return 1.0
    ratio = max(0.0, Pu) / phiPn0
    if ratio <= 0.2:
        return 1.0
    ratio = min(ratio, 0.99)
    return min(10.0, 1.0 / (1.0 - ratio))


def radial_capacity_ratio(Pu, M_val, curve):
    tol = 1e-9
    axis_tol = 1e-6
    moment_tol = 1e-6
    M_abs = abs(M_val)

    v = np.array([M_abs, Pu], dtype=float)
    norm_v = np.hypot(v[0], v[1])

    if norm_v <= tol:
        cap = max(curve.phiMn(0.0), 0.0)
        return 0.0, cap, 0.0, 0.0

    if abs(Pu) <= axis_tol and M_abs > moment_tol:
        phi_zero = max(curve.phiMn(0.0), 0.0)
        if phi_zero <= tol:
            return np.inf, 0.0, float('nan'), float('nan')
        ratio = M_abs / phi_zero
        return ratio, phi_zero, 0.0, math.copysign(phi_zero, M_val)

    if M_abs <= moment_tol:
        if Pu >= 0.0:
            axis_candidates = [p for p, m in zip(curve.P, curve.M) if p >= -1e-8 and abs(m) <= moment_tol]
        else:
            axis_candidates = [p for p, m in zip(curve.P, curve.M) if p <= 1e-8 and abs(m) <= moment_tol]
        if axis_candidates:
            Pu_cap_axis = max(axis_candidates) if Pu >= 0.0 else min(axis_candidates)
            cap_distance = abs(Pu_cap_axis)
            if cap_distance > tol:
                ratio = norm_v / cap_distance
                Pu_cap = Pu_cap_axis
                M_cap = math.copysign(0.0, M_val)
                return ratio, 0.0, Pu_cap, M_cap

    direction_positive = Pu >= 0.0

    raw_points = getattr(curve, 'poly_points', None)
    if raw_points is None:
        raw_points = list(zip(curve.P, np.maximum(curve.M, 0.0)))

    if direction_positive:
        points = [(float(p), float(m)) for p, m in raw_points if p >= -1e-6]
    else:
        points = [(float(p), float(m)) for p, m in raw_points if p <= 1e-6]

    if not points:
        points = [(float(p), float(m)) for p, m in raw_points]

    phi_zero = max(curve.phiMn(0.0), 0.0)
    has_zero = any(abs(p) <= 1e-8 for p, _ in points)
    if not has_zero:
        points.append((0.0, phi_zero))

    if direction_positive:
        points.sort(key=lambda pm: pm[0])
    else:
        points.sort(key=lambda pm: pm[0], reverse=True)

    filtered = []
    for p, m in points:
        if filtered and abs(p - filtered[-1][0]) <= 1e-8 and abs(m - filtered[-1][1]) <= 1e-8:
            continue
        filtered.append((p, m))

    if not filtered:
        return np.inf, 0.0, float('nan'), float('nan')

    F_values = [m * Pu - p * M_abs for p, m in filtered]

    best_P = None
    best_M = None

    for idx, F_val in enumerate(F_values):
        if abs(F_val) <= 1e-8:
            best_P, best_M = filtered[idx]
            break
        if idx + 1 >= len(filtered):
            continue
        next_F = F_values[idx + 1]
        if F_val == next_F:
            continue
        if F_val > 0.0 and next_F > 0.0:
            continue
        if F_val < 0.0 and next_F < 0.0:
            continue
        t = F_val / (F_val - next_F)
        t = np.clip(t, 0.0, 1.0)
        P_cap = filtered[idx][0] + t * (filtered[idx + 1][0] - filtered[idx][0])
        M_cap = filtered[idx][1] + t * (filtered[idx + 1][1] - filtered[idx][1])
        best_P, best_M = P_cap, max(M_cap, 0.0)
        break

    if best_P is None:
        plateau_adjusted = False
        if direction_positive and Pu > tol:
            plateau_P = getattr(curve, 'plateau_P', None)
            plateau_min = getattr(curve, 'plateau_Mmin', None)
            plateau_max = getattr(curve, 'plateau_Mmax', None)
            if (plateau_P is not None and plateau_min is not None and plateau_max is not None
                    and plateau_max > plateau_min + 1e-9):
                scale_plateau = plateau_P / max(Pu, tol)
                M_plateau = scale_plateau * M_abs
                if plateau_min - 1e-6 <= M_plateau <= plateau_max + 1e-6:
                    best_P = plateau_P
                    best_M = max(min(M_plateau, plateau_max), plateau_min)
                    plateau_adjusted = True
        if not plateau_adjusted:
            min_idx = int(np.argmin([abs(F) for F in F_values]))
            best_P, best_M = filtered[min_idx]

    if abs(Pu) > axis_tol:
        scale = best_P / Pu
    elif M_abs > moment_tol:
        scale = best_M / M_abs
    else:
        scale = 0.0

    if scale <= tol:
        return np.inf, 0.0, float('nan'), float('nan')

    ratio = 1.0 / scale
    M_cap_abs = best_M
    Pu_cap = best_P
    M_cap = math.copysign(M_cap_abs, M_val)
    return ratio, M_cap_abs, Pu_cap, M_cap


def evaluate_bresler(Pu, My, Mz, curve_y, curve_z, phiPn0):
    ratio_y, cap_y, Pu_cap_y, My_cap = radial_capacity_ratio(Pu, My, curve_y)
    ratio_z, cap_z, Pu_cap_z, Mz_cap = radial_capacity_ratio(Pu, Mz, curve_z)
    if not np.isfinite(ratio_y) or not np.isfinite(ratio_z):
        return np.inf, 1.0, cap_y, cap_z, ratio_y, ratio_z, (Pu_cap_y, My_cap), (Pu_cap_z, Mz_cap)
    alpha = bresler_alpha(Pu, phiPn0) if Pu >= 0.0 else 1.0
    demand = ratio_y**alpha + ratio_z**alpha
    return demand, alpha, cap_y, cap_z, ratio_y, ratio_z, (Pu_cap_y, My_cap), (Pu_cap_z, Mz_cap)

def bresler_boundary(Pu, curve_y, curve_z, phiPn0, npts=361):
    cap_y = curve_y.phiMn(Pu)
    cap_z = curve_z.phiMn(Pu)
    if cap_y <= 1e-6 or cap_z <= 1e-6:
        return None
    alpha = bresler_alpha(Pu, phiPn0) if Pu >= 0.0 else 1.0
    xs = np.linspace(-cap_y, cap_y, npts)
    base = np.clip(1.0 - np.power(np.abs(xs) / max(cap_y, 1e-9), alpha), 0.0, 1.0)
    ys = cap_z * np.power(base, 1.0 / alpha)
    return xs, ys, alpha

COLUMN_BAR_DIAMETER_OPTIONS = [0.0127, 0.015875, 0.01905, 0.0381]
COLUMN_BAR_COUNT_OPTIONS = [8, 12, 16, 20, 24, 28, 32, 36, 40]
COLUMN_SECTION_INCREMENT = 0.05
COLUMN_SECTION_MAX_STEPS = 20


def collect_column_demand_points(column_combo_forces, combos, elements=None):
    points = []
    if elements is None:
        iterable = column_combo_forces.items()
    else:
        iterable = ((ele, column_combo_forces[ele]) for ele in elements if ele in column_combo_forces)
    for ele, data in iterable:
        for end in ('i', 'j'):
            end_forces = data.get(end, {})
            for combo in combos:
                demand = end_forces.get(combo)
                if demand is None:
                    continue
                points.append({
                    'Pu': demand['Pu'],
                    'My': demand['My'],
                    'Mz': demand['Mz']
                })
    return points


def design_uniform_column_section(demand_points, initial_side,
                                  cover, stirrup_diam, bar_count_options, bar_diam_options):
    if not demand_points:
        raise ValueError('No se encontraron demandas para el diseño de columnas.')

    step = COLUMN_SECTION_INCREMENT
    tol = 1e-6
    min_side = ceil_to_step(0.30, step)
    start_side = ceil_to_step(max(initial_side, min_side), step)
    max_side = start_side + step * COLUMN_SECTION_MAX_STEPS

    bar_options = sorted(set(bar_diam_options))
    bar_counts = sorted(set(bar_count_options))
    candidate_pairs = []
    for idx, corner_diam in enumerate(bar_options):
        candidate_pairs.append((corner_diam, None))
        if idx + 1 < len(bar_options):
            next_diam = bar_options[idx + 1]
            candidate_pairs.append((corner_diam, next_diam))
            candidate_pairs.append((next_diam, corner_diam))

    def attempt_side(side):
        best = None
        best_metric = None
        for n_bars in bar_counts:
            for corner_diam, face_diam in candidate_pairs:
                try:
                    layout = build_column_rebar_layout(side, side, cover, stirrup_diam,
                                                      corner_diam, n_bars, face_diam)
                except ValueError:
                    continue
                As_total = sum(bar['area'] for bar in layout)
                Ag = side * side
                rho_long = As_total / Ag
                if rho_long < 0.01 - tol or rho_long > 0.025 + tol:
                    continue

                Pn0_nom_kN = 0.85 * fc_col * (Ag - As_total) * 1000.0 + fy_col * As_total * 1000.0
                phiPn0_ton = phi_axial_col * Pn0_nom_kN * kN_TO_TON
                Pn0_limit_ton = 0.8 * Pn0_nom_kN * kN_TO_TON

                uniaxial_y_curve = compute_uniaxial_curve('y', layout, side, side, Pn0_limit_ton)
                uniaxial_z_curve = compute_uniaxial_curve('z', layout, side, side, Pn0_limit_ton)
                moment_curve_y = MomentCapacityCurve(uniaxial_y_curve['phiPn'], uniaxial_y_curve['phiMn'], phiPn0_ton)
                moment_curve_z = MomentCapacityCurve(uniaxial_z_curve['phiPn'], uniaxial_z_curve['phiMn'], phiPn0_ton)

                phiPn_min = min(uniaxial_y_curve['phiPn'].min(), uniaxial_z_curve['phiPn'].min())
                phiPn_max = max(uniaxial_y_curve['phiPn'].max(), uniaxial_z_curve['phiPn'].max(), phiPn0_ton)

                all_ok = True
                max_ratio = 0.0
                for point in demand_points:
                    Pu = point['Pu']
                    My = point['My']
                    Mz = point['Mz']
                    if Pu > phiPn_max + tol or Pu < phiPn_min - tol:
                        all_ok = False
                        break
                    ratio, alpha, cap_y, cap_z, ratio_y, ratio_z, _, _ = evaluate_bresler(
                        Pu, My, Mz, moment_curve_y, moment_curve_z, phiPn0_ton)
                    if ratio > 1.0 + tol:
                        all_ok = False
                        break
                    if abs(My) > cap_y * (1.0 + tol) or abs(Mz) > cap_z * (1.0 + tol):
                        all_ok = False
                        break
                    max_ratio = max(max_ratio, ratio)

                if not all_ok:
                    continue

                face_use = face_diam if face_diam is not None else corner_diam
                metric = (
                    n_bars,
                    As_total,
                    -max_ratio,
                    corner_diam,
                    face_use
                )
                if best is None or metric < best_metric:
                    best_metric = metric
                    best = {
                        'side': side,
                        'n_bars': n_bars,
                        'bar_corners_diameter': corner_diam,
                        'bar_faces_diameter': face_use,
                        'layout': layout,
                        'uniaxial_y_curve': uniaxial_y_curve,
                        'uniaxial_z_curve': uniaxial_z_curve,
                        'moment_curve_y': moment_curve_y,
                        'moment_curve_z': moment_curve_z,
                        'phiPn0_ton': phiPn0_ton,
                        'Pn0_limit_ton': Pn0_limit_ton,
                        'As_total': As_total,
                        'rho_long': rho_long,
                        'max_ratio': max_ratio
                    }
        return best

    # Asegurar que existe una solución a partir del tamaño inicial (permitiendo incrementos)
    side = start_side
    best_design = None
    while side <= max_side + tol:
        candidate = attempt_side(side)
        if candidate is not None:
            best_design = candidate
            break
        side += step

    if best_design is None:
        raise RuntimeError('No se pudo encontrar una sección de columna uniforme que cumpla los requisitos.')

    # Intentar reducir progresivamente la sección hasta alcanzar el mínimo que cumple
    next_side = floor_to_step(best_design['side'] - step, step)
    while next_side >= min_side - tol:
        candidate = attempt_side(next_side)
        if candidate is None:
            break
        best_design = candidate
        next_side = floor_to_step(candidate['side'] - step, step)

    return best_design


def describe_column_rebar(layout):
    groups = classify_column_rebar_layout(layout)

    def describe_group(bars, template):
        if not bars:
            return []
        counts = {}
        for bar in bars:
            diam = bar['diameter']
            counts[diam] = counts.get(diam, 0) + 1
        parts_local = []
        for diam in sorted(counts.keys()):
            parts_local.append(template.format(
                count=counts[diam],
                label=bar_diameter_label(diam)
            ))
        return parts_local

    parts = []
    parts.extend(describe_group(groups['esquinas'], "{count} barras Ø{label} en esquinas"))
    parts.extend(describe_group(groups['caras_y'], "{count} barras Ø{label} centradas en caras ∥ Y"))
    parts.extend(describe_group(groups['caras_x'], "{count} barras Ø{label} centradas en caras ∥ X"))
    parts.extend(describe_group(groups['interiores'], "{count} barras Ø{label} interiores"))

    if parts:
        body = human_join(parts)
    else:
        body = summarize_rebar_counts(layout)

    counts_by_diam = {}
    for bar in layout:
        diam = bar['diameter']
        counts_by_diam[diam] = counts_by_diam.get(diam, 0) + 1
    counts_summary = " + ".join(
        f"{counts_by_diam[diam]}Ø{bar_diameter_label(diam)}" for diam in sorted(counts_by_diam.keys())
    )
    return groups, body, counts_summary, sorted(counts_by_diam.keys())


def group_columns_by_level(columns, levels_per_group=3):
    if not columns:
        return []
    max_level = max(iz for (_, _, _, iz) in columns)
    groups = []
    start = 1
    while start <= max_level:
        end = min(start + levels_per_group - 1, max_level)
        groups.append({'level_range': (start, end), 'columns': []})
        start = end + 1
    for ele, ix, iy, iz in columns:
        for group in groups:
            low, high = group['level_range']
            if low <= iz <= high:
                group['columns'].append(ele)
                break
    return groups


initial_column_side = min(sec_col_b, sec_col_h)
column_level_groups = group_columns_by_level(columns, levels_per_group=3)
column_design_groups = []
column_design_by_element = {}
column_side_by_level = {}


def build_interaction_plot_data(uniaxial_curve):
    order = np.argsort(uniaxial_curve['Pn'])
    P_sorted = uniaxial_curve['Pn'][order]
    phiP_sorted = uniaxial_curve['phiPn'][order]
    M_sorted = uniaxial_curve['Mn'][order]
    phiM_sorted = uniaxial_curve['phiMn'][order]
    P_vals, phiP_vals, M_vals, phiM_vals = [], [], [], []
    last_P = None
    for P_val, phiP_val, M_val, phiM_val in zip(P_sorted, phiP_sorted, M_sorted, phiM_sorted):
        if last_P is not None and abs(P_val - last_P) < 1e-4:
            if abs(M_val) > abs(M_vals[-1]):
                M_vals[-1] = M_val
                phiM_vals[-1] = phiM_val
            if abs(phiP_val) > abs(phiP_vals[-1]):
                phiP_vals[-1] = phiP_val
        else:
            P_vals.append(P_val)
            phiP_vals.append(phiP_val)
            M_vals.append(M_val)
            phiM_vals.append(phiM_val)
            last_P = P_val
    return {
        'Pn': np.array(P_vals, dtype=float),
        'phiPn': np.array(phiP_vals, dtype=float),
        'Mn': np.array(M_vals, dtype=float),
        'phiMn': np.array(phiM_vals, dtype=float)
    }

for group in column_level_groups:
    level_low, level_high = group['level_range']
    subset = group['columns']
    if not subset:
        continue
    demand_points = collect_column_demand_points(column_combo_forces, combo_names, subset)
    design = design_uniform_column_section(demand_points, initial_column_side,
                                           clear_cover_col, stirrup_diam,
                                           COLUMN_BAR_COUNT_OPTIONS, COLUMN_BAR_DIAMETER_OPTIONS)
    layout = design['layout']
    groups_layout, rebar_body, rebar_counts_summary, bar_diameters = describe_column_rebar(layout)
    moment_curve_y = design['moment_curve_y']
    moment_curve_z = design['moment_curve_z']
    interaction_data = {
        'y': build_interaction_plot_data(design['uniaxial_y_curve']),
        'z': build_interaction_plot_data(design['uniaxial_z_curve'])
    }
    design_info = {
        **design,
        'level_low': level_low,
        'level_high': level_high,
        'rebar_groups': groups_layout,
        'rebar_description': rebar_body,
        'rebar_summary': rebar_counts_summary,
        'bar_diameters': bar_diameters,
        'interaction_plot': interaction_data,
        'plateau': {
            'y': moment_curve_y.plateau_segment(),
            'z': moment_curve_z.plateau_segment(),
        }
    }
    column_design_groups.append({
        'levels': (level_low, level_high),
        'columns': subset,
        'design': design_info
    })
    for level in range(level_low, level_high + 1):
        column_side_by_level[level] = design['side']
    for ele in subset:
        column_design_by_element[ele] = design_info

if not column_design_groups:
    raise RuntimeError('No se pudo determinar el diseño de columnas por niveles.')

def column_group_label(level_low, level_high):
    if level_low == level_high:
        return f"nivel {level_low}"
    return f"niveles {level_low}–{level_high}"


COLUMN_STEEL_NOTE_LINES = []
COLUMN_SECTION_SUMMARY_PARTS = []
COLUMN_PHI_SUMMARY_PARTS = []
for entry in column_design_groups:
    design = entry['design']
    levels = entry['levels']
    note = (
        f"{column_group_label(*levels).capitalize()}: sección {design['side']:.2f}×{design['side']:.2f} m "
        f"con {design['rebar_description']}"
    )
    COLUMN_STEEL_NOTE_LINES.append(note)
    COLUMN_SECTION_SUMMARY_PARTS.append(
        f"{column_group_label(*levels).capitalize()}: {design['side']:.2f}×{design['side']:.2f} m"
    )
    COLUMN_PHI_SUMMARY_PARTS.append(
        f"{column_group_label(*levels).capitalize()}: ϕPn0={design['phiPn0_ton']:.2f} t"
    )

COLUMN_STEEL_NOTE = (
    "Configuración replicada de Prueba35.py: "
    + " | ".join(COLUMN_STEEL_NOTE_LINES)
    + ". Distribución optimizada y simétrica en cada tramo."
)

COLUMN_STEEL_NOTE_SHORT = (
    "Refuerzo Prueba35.py: "
    + " | ".join(COLUMN_STEEL_NOTE_LINES)
    + "."
)

COLUMN_SECTION_SUMMARY = " · ".join(COLUMN_SECTION_SUMMARY_PARTS)
COLUMN_PHI_SUMMARY = " · ".join(COLUMN_PHI_SUMMARY_PARTS)

column_design_checks = {}
for (ele, ix, iy, iz) in columns_by_level:
    meta = column_combo_forces[ele]['meta']
    design = column_design_by_element[ele]
    moment_curve_y = design['moment_curve_y']
    moment_curve_z = design['moment_curve_z']
    phiPn0_ton = design['phiPn0_ton']
    column_design_checks[ele] = {
        'i': {},
        'j': {},
        'summary': {
            'ix': ix,
            'iy': iy,
            'nivel_top': iz,
            'nivel_bottom': iz-1,
            'z_top': meta['z_top'],
            'z_bottom': meta['z_bottom'],
            'side': design['side'],
            'As_total': design['As_total'],
            'rho_long': design['rho_long'],
            'n_bars': design['n_bars'],
            'phiPn0': phiPn0_ton,
            'Pn0_limit': design['Pn0_limit_ton'],
            'level_group': (design['level_low'], design['level_high']),
            'rebar_description': design['rebar_description'],
            'rebar_summary': design['rebar_summary'],
            'bar_diameters': design['bar_diameters'],
            'bar_corners_diameter': design['bar_corners_diameter'],
            'bar_faces_diameter': design['bar_faces_diameter'],
            'interaction_plot': design['interaction_plot'],
            'plateau': design.get('plateau', {})
        }
    }
    ratios_all = []
    ratios_y_all = []
    ratios_z_all = []
    Pu_all = []
    for end in ('i', 'j'):
        for combo in combo_names:
            data = column_combo_forces[ele][end][combo]
            Pu = data['Pu']
            My = data['My']
            Mz = data['Mz']
            demand, alpha, cap_y, cap_z, ratio_y, ratio_z, cap_point_y, cap_point_z = evaluate_bresler(
                Pu, My, Mz, moment_curve_y, moment_curve_z, phiPn0_ton)
            check = {
                'Pu': Pu,
                'My': My,
                'Mz': Mz,
                'ratio': demand,
                'cap_y': cap_y,
                'cap_z': cap_z,
                'ratio_y': ratio_y,
                'ratio_z': ratio_z,
                'phi_point_y': cap_point_y,
                'phi_point_z': cap_point_z,
                'alpha': alpha
            }
            column_design_checks[ele][end][combo] = check
            ratios_all.append(demand)
            ratios_y_all.append(ratio_y)
            ratios_z_all.append(ratio_z)
            Pu_all.append(Pu)
    column_design_checks[ele]['summary']['max_ratio'] = max(ratios_all) if ratios_all else 0.0
    column_design_checks[ele]['summary']['max_ratio_y'] = max(ratios_y_all) if ratios_y_all else 0.0
    column_design_checks[ele]['summary']['max_ratio_z'] = max(ratios_z_all) if ratios_z_all else 0.0
    column_design_checks[ele]['summary']['ok'] = column_design_checks[ele]['summary']['max_ratio'] <= 1.0 + 1e-6
    if Pu_all:
        column_design_checks[ele]['summary']['Pu_max_abs'] = max(abs(p) for p in Pu_all)
        column_design_checks[ele]['summary']['Pu_max_comp'] = max((p for p in Pu_all if p >= 0.0), default=0.0)
        column_design_checks[ele]['summary']['Pu_min_ten'] = min((p for p in Pu_all if p <= 0.0), default=0.0)
    else:
        column_design_checks[ele]['summary']['Pu_max_abs'] = 0.0
        column_design_checks[ele]['summary']['Pu_max_comp'] = 0.0
        column_design_checks[ele]['summary']['Pu_min_ten'] = 0.0
    unique_P = sorted({float(f"{p:.4f}") for p in Pu_all})
    max_levels = 8
    if len(unique_P) > max_levels:
        idxs = np.linspace(0, len(unique_P)-1, max_levels).astype(int)
        Pu_plot_levels = [unique_P[i] for i in idxs]
    else:
        Pu_plot_levels = unique_P
    if 0.0 not in Pu_plot_levels:
        Pu_plot_levels.append(0.0)
    column_design_checks[ele]['summary']['Pu_levels'] = sorted(Pu_plot_levels)

displacement_envelope = {iz: {'UX_max': -np.inf, 'UX_min': np.inf,
                              'UY_max': -np.inf, 'UY_min': np.inf,
                              'RZ_max': -np.inf, 'RZ_min': np.inf}
                         for iz in floor_master}
columns_sorted_by_ratio = sorted(columns, key=lambda item: column_design_checks[item[0]]['summary']['max_ratio'], reverse=True)
column_summary_rows = []
for (ele, ix, iy, iz) in columns_sorted_by_ratio:
    summary = column_design_checks[ele]['summary']
    status = 'OK' if summary['ok'] else 'No cumple'
    column_summary_rows.append([
        ele,
        f"({ix},{iy})",
        f"{summary['nivel_bottom']} -> {summary['nivel_top']}",
        f"{summary['Pu_max_comp']:.2f}",
        f"{summary['Pu_min_ten']:.2f}",
        f"{summary['max_ratio']:.2f}",
        status
    ])
drift_envelope = {iz: {'X_max': -np.inf, 'X_min': np.inf,
                       'Y_max': -np.inf, 'Y_min': np.inf}
                  for iz in floor_master}
for combo, factors in load_combinations.items():
    combo_disp = {}
    for iz, node in floor_master.items():
        ux = uy = rz = 0.0
        for case, coef in factors.items():
            disp = case_results[case]['displacements'][iz]
            ux += coef * disp[0]
            uy += coef * disp[1]
            rz += coef * disp[2]
        combo_disp[iz] = (ux, uy, rz)
        env = displacement_envelope[iz]
        env['UX_max'] = max(env['UX_max'], ux)
        env['UX_min'] = min(env['UX_min'], ux)
        env['UY_max'] = max(env['UY_max'], uy)
        env['UY_min'] = min(env['UY_min'], uy)
        env['RZ_max'] = max(env['RZ_max'], rz)
        env['RZ_min'] = min(env['RZ_min'], rz)
    for iz in range(1, nz+1):
        ux_curr, uy_curr, _ = combo_disp.get(iz, (0.0, 0.0, 0.0))
        ux_prev, uy_prev = (0.0, 0.0) if iz == 1 else combo_disp.get(iz-1, (0.0, 0.0, 0.0))[:2]
        drift_x = ux_curr - ux_prev
        drift_y = uy_curr - uy_prev
        env = drift_envelope[iz]
        env['X_max'] = max(env['X_max'], drift_x)
        env['X_min'] = min(env['X_min'], drift_x)
        env['Y_max'] = max(env['Y_max'], drift_y)
        env['Y_min'] = min(env['Y_min'], drift_y)

dominant_mode_X = max(modal_X['modes'], key=lambda m: m['mass_ratio']) if modal_X['modes'] else None
dominant_mode_Y = max(modal_Y['modes'], key=lambda m: m['mass_ratio']) if modal_Y['modes'] else None
mass_ratio_total_X = modal_X['modes'][-1]['cum_mass_ratio'] if modal_X['modes'] else 0.0
mass_ratio_total_Y = modal_Y['modes'][-1]['cum_mass_ratio'] if modal_Y['modes'] else 0.0

roof_disp_x = roof_disp_y = 0.0
if nz in displacement_envelope:
    roof_data = displacement_envelope[nz]
    roof_disp_x = max(abs(roof_data['UX_max']), abs(roof_data['UX_min']))
    roof_disp_y = max(abs(roof_data['UY_max']), abs(roof_data['UY_min']))

max_drift_ratio_x = 0.0
max_drift_ratio_y = 0.0
drift_profile = {
    'floors': np.arange(1, nz+1, dtype=int) if nz > 0 else np.array([], dtype=int),
    'ratio_x': [],
    'ratio_y': []
}
for iz in range(1, nz+1):
    env = drift_envelope[iz]
    drift_x = max(abs(env['X_max']), abs(env['X_min']))
    drift_y = max(abs(env['Y_max']), abs(env['Y_min']))
    h = H[iz-1] if (iz-1) < len(H) and len(H) > 0 else 1.0
    if h <= 0:
        ratio_x = ratio_y = 0.0
    else:
        ratio_x = drift_x / h
        ratio_y = drift_y / h
        max_drift_ratio_x = max(max_drift_ratio_x, ratio_x)
        max_drift_ratio_y = max(max_drift_ratio_y, ratio_y)
    drift_profile['ratio_x'].append(ratio_x)
    drift_profile['ratio_y'].append(ratio_y)
drift_profile['ratio_x'] = np.array(drift_profile['ratio_x'], dtype=float)
drift_profile['ratio_y'] = np.array(drift_profile['ratio_y'], dtype=float)

base_shear_x = abs(modal_X['base_shear'])
base_shear_y = abs(modal_Y['base_shear'])
base_shear_x_ton = base_shear_x * kN_TO_TON
base_shear_y_ton = base_shear_y * kN_TO_TON
total_mass_ton = total_weight / g_grav
total_weight_ton = total_weight * kN_TO_TON
Tx1 = dominant_mode_X['T'] if dominant_mode_X else 0.0
Ty1 = dominant_mode_Y['T'] if dominant_mode_Y else 0.0
mode_mass_x_pct = dominant_mode_X['mass_ratio'] * 100 if dominant_mode_X else 0.0
mode_mass_y_pct = dominant_mode_Y['mass_ratio'] * 100 if dominant_mode_Y else 0.0
mass_ratio_total_X_pct = mass_ratio_total_X * 100
mass_ratio_total_Y_pct = mass_ratio_total_Y * 100
drift_x_pct = max_drift_ratio_x * 100
drift_y_pct = max_drift_ratio_y * 100
roof_disp_x_m = roof_disp_x
roof_disp_y_m = roof_disp_y
# ------------------ RESPUESTAS Y DIBUJOS ------------------
def ele_nodes(e):
    i,j = ops.eleNodes(e)
    xi,yi,zi = ops.nodeCoord(i); xj,yj,zj = ops.nodeCoord(j)
    L = ((xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2)**0.5
    return (i,j), (np.array([xi,yi,zi]), np.array([xj,yj,zj])), L

def local_end_forces(e):
    f = np.array(ops.eleResponse(e,'localForce'), dtype=float)
    if f.size == 12: return f
    if f.size == 6:  return np.concatenate([f, -f])
    g = np.array(ops.eleResponse(e,'force'), dtype=float)
    return g if g.size==12 else np.concatenate([g[:6], -g[:6]])

def vertical_fill(ax, x, y, base=0, density=250, color='k', lw=0.7):
    step = max(1, int(len(x)/density))
    segs = [[(x[i], base), (x[i], y[i])] for i in range(0, len(x), step)]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=0.9))


def column_diagram_envelope(ax, z, pos_curve, neg_curve, color, xlabel, set_ylabel=True):
    z = np.asarray(z)
    pos_curve = np.asarray(pos_curve, dtype=float)
    neg_curve = np.asarray(neg_curve, dtype=float)
    pos_curve = np.where(np.isfinite(pos_curve), pos_curve, 0.0)
    neg_curve = np.where(np.isfinite(neg_curve), neg_curve, 0.0)
    pos_curve = np.maximum(pos_curve, 0.0)
    neg_curve = np.minimum(neg_curve, 0.0)

    ax.fill_betweenx(z, 0.0, pos_curve, color=color, alpha=0.35)
    ax.fill_betweenx(z, 0.0, neg_curve, color=color, alpha=0.35)
    ax.plot(pos_curve, z, color=color, lw=2.0)
    ax.plot(neg_curve, z, color=color, lw=2.0)
    ax.axvline(0.0, color='0.25', lw=1.2)

    span = max(np.max(np.abs(pos_curve)), np.max(np.abs(neg_curve)), 1e-6)
    ax.set_xlim(-1.15 * span, 1.15 * span)
    ax.set_ylim(z[0], z[-1])
    if set_ylabel:
        ax.set_ylabel('z local [m]')
    ax.set_xlabel(xlabel)
    ax.grid(True, ls=':', alpha=0.35)

# -------- Planta por nivel (columnas rectangulares + vigas banda) ---------
def draw_plan_level(ax, level_index, highlight=None):
    level_z = Z[level_index]
    ax.set_title(f"Planta — Nivel z = {level_z:g} m", fontsize=10, weight='bold')
    side = column_side_by_level.get(level_index, sec_col_plan[0])
    bx = by = side
    # Columnas (rectángulos)
    for iy in range(ny+1):
        for ix in range(nx+1):
            cx, cy = X[ix], Y[iy]
            rect = np.array([[cx-bx/2, cy-by/2],
                             [cx+bx/2, cy-by/2],
                             [cx+bx/2, cy+by/2],
                             [cx-bx/2, cy+by/2],
                             [cx-bx/2, cy-by/2]])
            ax.plot(rect[:,0], rect[:,1], color='k', lw=1.5)
            ax.fill(rect[:,0], rect[:,1], color='0.85', zorder=1)
    # Vigas X
    for (ele, iz, iy, ix) in beamsX:
        if Z[iz] != level_z: continue
        x0, x1 = X[ix], X[ix+1]; y = Y[iy]; w = beam_plan_width
        poly = np.array([[x0, y - w/2],[x1, y - w/2],[x1, y + w/2],[x0, y + w/2],[x0, y - w/2]])
        col = 'tab:blue' if highlight==ele else '0.25'
        ax.plot(poly[:,0], poly[:,1], color=col, lw=1.4)
        ax.fill(poly[:,0], poly[:,1], color=(0.7,0.8,1.0) if highlight==ele else '0.9', zorder=0)
    # Vigas Y
    for (ele, iz, ix, iy) in beamsY:
        if Z[iz] != level_z: continue
        y0, y1 = Y[iy], Y[iy+1]; x = X[ix]; w = beam_plan_width
        poly = np.array([[x - w/2, y0],[x - w/2, y1],[x + w/2, y1],[x + w/2, y0],[x - w/2, y0]])
        col = 'tab:blue' if highlight==ele else '0.25'
        ax.plot(poly[:,0], poly[:,1], color=col, lw=1.4)
        ax.fill(poly[:,0], poly[:,1], color=(0.7,0.8,1.0) if highlight==ele else '0.9', zorder=0)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.5, X[-1]+0.5); ax.set_ylim(-0.5, Y[-1]+0.5)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.grid(True, ls=':', alpha=0.35)

# ------------------ 3D con prismas -----------------------
def orthonormal_basis_along(p, q, up_hint=np.array([0,0,1.0])):
    ex = q - p; L = np.linalg.norm(ex)
    if L == 0: return None, None, None, 0.0
    ex = ex / L
    uz = up_hint / np.linalg.norm(up_hint)
    if abs(np.dot(ex, uz)) > 0.98:
        uz = np.array([1,0,0], dtype=float)
    ey = np.cross(uz, ex); n = np.linalg.norm(ey)
    if n < 1e-8: ey = np.array([0,1,0]); n = 1.0
    ey = ey / n
    uz = np.cross(ex, ey)
    uz = uz / np.linalg.norm(uz)
    return ex, ey, uz, L

def prism_faces(p, q, half_y, half_z, up_hint=np.array([0,0,1.0])):
    ex, ey, uz, L = orthonormal_basis_along(p, q, up_hint)
    if L == 0: return []
    c = []
    for end in [p, q]:
        c.append(end + (-half_y)*ey + (-half_z)*uz)
        c.append(end + (+half_y)*ey + (-half_z)*uz)
        c.append(end + (+half_y)*ey + (+half_z)*uz)
        c.append(end + (-half_y)*ey + (+half_z)*uz)
    faces = [
        [c[0],c[1],c[2],c[3]],
        [c[4],c[5],c[6],c[7]],
        [c[0],c[1],c[5],c[4]],
        [c[1],c[2],c[6],c[5]],
        [c[2],c[3],c[7],c[6]],
        [c[3],c[0],c[4],c[7]],
    ]
    return faces

def draw_frame_3d(ax, highlight=None):
    # Columnas
    for (ele, ix, iy, iz) in columns_by_level:
        (_, (p,q), _) = ele_nodes(ele)
        design = column_design_by_element.get(ele)
        side = design['side'] if design else sec_col_b
        faces = prism_faces(p, q, side/2, side/2, up_hint=np.array([1,0,0]))
        col = (0.6,0.6,0.6) if highlight!=ele else (1.0,0.8,0.2)
        pc = Poly3DCollection(faces, facecolors=col, edgecolors='k', linewidths=0.6, alpha=0.95)
        ax.add_collection3d(pc)
    # Vigas
    for (ele, iz, iy, ix) in beamsX:
        (_, (p,q), _) = ele_nodes(ele)
        section = beam_section_by_element.get(ele, {'b': sec_beam_b, 'h': sec_beam_h})
        faces = prism_faces(p, q, section['b']/2, section['h']/2, up_hint=np.array([0,0,1.0]))
        col = (0.75,0.8,1.0) if highlight!=ele else (0.26,0.52,1.0)
        pc = Poly3DCollection(faces, facecolors=col, edgecolors='k', linewidths=0.5, alpha=0.95)
        ax.add_collection3d(pc)
    for (ele, iz, ix, iy) in beamsY:
        (_, (p,q), _) = ele_nodes(ele)
        section = beam_section_by_element.get(ele, {'b': sec_beam_b, 'h': sec_beam_h})
        faces = prism_faces(p, q, section['b']/2, section['h']/2, up_hint=np.array([0,0,1.0]))
        col = (0.75,0.8,1.0) if highlight!=ele else (0.26,0.52,1.0)
        pc = Poly3DCollection(faces, facecolors=col, edgecolors='k', linewidths=0.5, alpha=0.95)
        ax.add_collection3d(pc)

    ax.set_box_aspect([X[-1], Y[-1], Z[-1]])
    ax.set_xlim(-0.6, X[-1]+0.6); ax.set_ylim(-0.6, Y[-1]+0.6); ax.set_zlim(-0.2, Z[-1]+0.8)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')

# 3D con flechas de carga (longitud proporcional a w)
def draw_frame_3d_with_loads(ax):
    draw_frame_3d(ax)
    if not w_total_map:
        ax.set_title('3D (sin cargas gravitacionales)', fontsize=11, weight='bold')
        return
    wmax = max(w_total_map.values())
    if wmax <= 0:
        ax.set_title('3D (sin cargas gravitacionales)', fontsize=11, weight='bold')
        return

    def add_quivers(p, q, w, n_arrows=7):
        xs = np.linspace(p[0], q[0], n_arrows+2)[1:-1]
        ys = np.linspace(p[1], q[1], n_arrows+2)[1:-1]
        zs = np.linspace(p[2], q[2], n_arrows+2)[1:-1]
        L = 0.8 * max(0.25, (w / wmax))  # flecha proporcional con limite visual
        for x, y, z in zip(xs, ys, zs):
            ax.quiver(x, y, z, 0, 0, -L, arrow_length_ratio=0.25, color='k', linewidth=1.2)
    for (ele, iz, iy, ix) in beamsX:
        add_quivers(*ele_nodes(ele)[1], w_total_map.get(ele, 0.0) * kN_TO_TON)
    for (ele, iz, ix, iy) in beamsY:
        add_quivers(*ele_nodes(ele)[1], w_total_map.get(ele, 0.0) * kN_TO_TON)
    ax.set_title('3D (cargas gravitacionales)', fontsize=11, weight='bold')

# ------------------ Páginas de diagramas ------------------
def page_beam(ele, tag_txt, env):
    (_, (p, q), L) = ele_nodes(ele)
    x = env['x']
    V_pos = env['V_pos']
    V_neg = env['V_neg']
    M_pos = env['M_pos']
    M_neg = env['M_neg']
    design = beam_design_results.get(ele)
    section_h = design['h'] if design else sec_beam_h

    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.0], hspace=0.35)

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.set_title(f'Viga - {tag_txt}', fontsize=12, weight='bold')
    draw_frame_3d(ax3d, highlight=ele)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_title('Cortante $V_z$ - envolvente', fontsize=11)
    ax1.add_patch(plt.Rectangle((0, -section_h/2), L, section_h, ec='k', fc='0.9', lw=1.0))
    ax1.axhline(0, color='k', lw=0.8)
    ax1.plot(x, V_pos, color='tab:blue', lw=2.0)
    ax1.plot(x, V_neg, color='tab:blue', lw=2.0)
    vertical_fill(ax1, x, V_pos, color='tab:blue')
    vertical_fill(ax1, x, V_neg, color='tab:blue')
    Vy = max(np.max(np.abs(V_pos)), np.max(np.abs(V_neg)), 1e-6)
    ax1.set_xlim(-0.02 * L, 1.02 * L)
    ax1.set_ylim(-1.25 * Vy, 1.25 * Vy)
    ax1.set_xlabel('x local [m]')
    ax1.set_ylabel('$V_z$ [t]')
    ax1.grid(True, ls=':', alpha=0.35)

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_title('Momento $M_y$ - envolvente', fontsize=11)
    ax2.add_patch(plt.Rectangle((0, -section_h/2), L, section_h, ec='k', fc='0.9', lw=1.0))
    ax2.axhline(0, color='k', lw=0.8)
    ax2.plot(x, M_pos, color='tab:red', lw=2.0)
    ax2.plot(x, M_neg, color='tab:red', lw=2.0)
    vertical_fill(ax2, x, M_pos, color='tab:red')
    vertical_fill(ax2, x, M_neg, color='tab:red')
    Myabs = max(np.max(np.abs(M_pos)), np.max(np.abs(M_neg)), 1e-6)
    ax2.set_xlim(-0.02 * L, 1.02 * L)
    ax2.set_ylim(-1.25 * Myabs, 1.25 * Myabs)
    ax2.set_xlabel('x local [m]')
    ax2.set_ylabel('$M_y$ [t-m]')
    ax2.grid(True, ls=':', alpha=0.35)

    Vmax = np.max(V_pos)
    Vmin = np.min(V_neg)
    Mmax = np.max(M_pos)
    Mmin = np.min(M_neg)
    if design:
        section_info = f"Sección b={design['b']:.2f} m × h={design['h']:.2f} m ({design['pair_label']})"
    else:
        section_info = f"Sección b={sec_beam_b:.2f} m × h={sec_beam_h:.2f} m"
    footer = (f"L={L:.3f}  Vmax={Vmax:.3f} t  Vmin={Vmin:.3f} t  "
              f"Mmax={Mmax:.3f} t-m  Mmin={Mmin:.3f} t-m  {section_info}")
    fig.subplots_adjust(left=PAGE_LEFT, right=PAGE_RIGHT,
                        top=PAGE_TOP - 0.03, bottom=PAGE_BOTTOM + 0.08)
    return fig, footer


def _beam_layout_from_option(design_face):
    opt = design_face['option']
    if opt is None:
        return {'count': 0, 'diam': beam_bar_sizes['N6']}
    return {'count': opt['count'], 'diam': opt['diam']}


def draw_beam_design_block(fig, slot_spec, ele, tag_txt):
    design = beam_design_results[ele]
    bottom = design['bottom']
    top = design['top']

    block = slot_spec.subgridspec(3, 1, height_ratios=[0.46, 0.18, 0.36], hspace=0.16)

    ax_section = fig.add_subplot(block[0, 0])
    plot_beam_section(ax_section, design['b'], design['h'],
                      clear_cover_beam, stirrup_diam_beam,
                      _beam_layout_from_option(bottom), _beam_layout_from_option(top))
    ax_section.set_anchor('C')
    ax_section.set_title(tag_txt, fontsize=11, weight='bold', pad=10)

    ax_info = fig.add_subplot(block[1, 0])
    ax_info.axis('off')
    ax_info.set_anchor('N')
    info_lines = [
        f"fc'={fc_beam_kgf_cm2:.1f} kg/cm², fy={fy_beam_kgf_cm2:.0f} kg/cm², ϕ={phi_flex_beam:.2f}",
        f"Sección {design['b']:.2f}×{design['h']:.2f} m ({design['pair_label']})",
        f"Recubrimiento={clear_cover_beam*1000:.0f} mm",
        f"Mu(+)= {design['Mu_pos']:.2f} t-m  Mu(-)= {design['Mu_neg']:.2f} t-m",
        f"Estado global: {'OK' if design['ok'] else 'No cumple'}"
    ]
    if not design.get('section_ok', True):
        info_lines.append('⚠️ Revisar selección de sección: el grupo por niveles no satisface completamente la demanda crítica.')
    if design['needs_resize']:
        info_lines.append('⚠️ Requiere revisar sección o combinación de refuerzo (ϕMn < Mu).')
    info_lines.append(BEAM_STEEL_NOTE_SHORT)
    ax_info.text(0.5, 0.96, "\n".join(info_lines), ha='center', va='top', fontsize=9.4,
               transform=ax_info.transAxes, wrap=True)

    def format_face_row(label, face_design):
        opt = face_design['option']
        As_req_cm2 = face_design['As_req'] * 1e4
        As_min_cm2 = face_design['As_min'] * 1e4
        As_prov_cm2 = face_design.get('As_prov', 0.0) * 1e4
        phiMn = face_design['phiMn_ton_m']
        Mu = face_design['Mu_ton_m']
        ratio = face_design['ratio'] if Mu > 1e-6 else (As_req_cm2 / max(As_prov_cm2, 1e-6))
        ref = opt['label'] if opt else '—'
        estado = 'OK' if face_design['ok'] else 'No cumple'
        return [
            label,
            f"{Mu:.2f}",
            f"{phiMn:.2f}",
            f"{As_req_cm2:.2f}",
            f"{As_min_cm2:.2f}",
            f"{As_prov_cm2:.2f}",
            ref,
            f"{ratio:.2f}",
            estado
        ]

    ax_table = fig.add_subplot(block[2, 0])
    ax_table.axis('off')
    table_cols = ['Cara', 'Mu [t-m]', 'ϕMn [t-m]', 'As req [cm²]', 'As min [cm²]',
                  'As prov [cm²]', 'Refuerzo', 'Demanda', 'Estado']
    table_rows = [
        format_face_row('Inferior (+)', bottom),
        format_face_row('Superior (-)', top)
    ]
    col_widths = [0.12, 0.11, 0.15, 0.11, 0.10, 0.10, 0.12, 0.07, 0.12]
    table = ax_table.table(cellText=table_rows, colLabels=table_cols,
                           colWidths=col_widths, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(0.98, 1.15)
    for r, row in enumerate(table_rows):
        if row[-1] != 'OK':
            for c in range(len(table_cols)):
                table[(r+1, c)].set_facecolor('#f8d7da')
    ax_table.text(0.5, -TABLE_CAPTION_GAP,
                  table_caption(('beam_faces', ele)),
                  transform=ax_table.transAxes, ha='center', va='top', fontsize=9)


def build_beam_design_page(entries):
    rows = len(entries)
    if rows == 0:
        return None
    fig = plt.figure(figsize=(8.27, 11.69))
    outer = fig.add_gridspec(rows, 1, height_ratios=[1] * rows, hspace=0.32)
    for idx, (ele, tag_txt) in enumerate(entries):
        slot = outer[idx, 0]
        draw_beam_design_block(fig, slot, ele, tag_txt)
    apply_page_margins(fig)
    return fig


def prepare_caption_registry():
    _reset_caption_registry()
    if beam_summary_rows:
        register_table(('beam_summary', 'general'), 'Resumen de diseño de vigas')
    for (_, ele, tag_txt) in ordered_beams:
        register_table(('beam_faces', ele), f'Revisión de refuerzo por cara · {tag_txt}')
    if column_summary_rows:
        register_table(('column_summary', 'general'), 'Resumen de demandas y verificación de columnas')
    for (ele, ix, iy, iz) in columns_by_level:
        tag_txt = f"(x={ix}, y={iy}, piso {iz}, ele={ele})"
        register_table(('column_combos', ele), f'Combinaciones de carga y verificaciones · Columna {tag_txt}')
        register_figure(('column_interaction_my', ele), f'Interacción My - Pu · Columna {tag_txt}')
        register_figure(('column_interaction_mz', ele), f'Interacción Mz - Pu · Columna {tag_txt}')


def page_column(ele, tag_txt, env):
    (_, (p, q), L) = ele_nodes(ele)
    z = env['z']
    My_pos = env['My_pos']
    My_neg = env['My_neg']
    Mz_pos = env['Mz_pos']
    Mz_neg = env['Mz_neg']

    Vy_pos = env['Vy_pos']
    Vy_neg = env['Vy_neg']
    Vz_pos = env['Vz_pos']
    Vz_neg = env['Vz_neg']

    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], hspace=0.32, wspace=0.28)

    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    ax3d.set_title(f'Columna - {tag_txt}', fontsize=12, weight='bold')
    draw_frame_3d(ax3d, highlight=ele)

    ax_vy = fig.add_subplot(gs[1, 0])
    ax_vy.set_title('Cortante $V_y$ - envolvente', fontsize=11, weight='bold')
    column_diagram_envelope(ax_vy, z, Vy_pos, Vy_neg, color='tab:orange', xlabel='$V_y$ [t]')

    ax_vz = fig.add_subplot(gs[1, 1])
    ax_vz.set_title('Cortante $V_z$ - envolvente', fontsize=11, weight='bold')
    column_diagram_envelope(ax_vz, z, Vz_pos, Vz_neg, color='tab:blue', xlabel='$V_z$ [t]', set_ylabel=False)

    ax_my = fig.add_subplot(gs[2, 0])
    ax_my.set_title('Momento $M_y$ - envolvente', fontsize=11, weight='bold')
    column_diagram_envelope(ax_my, z, My_pos, My_neg, color='tab:red', xlabel='$M_y$ [t·m]')

    ax_mz = fig.add_subplot(gs[2, 1])
    ax_mz.set_title('Momento $M_z$ - envolvente', fontsize=11, weight='bold')
    column_diagram_envelope(ax_mz, z, Mz_pos, Mz_neg, color='tab:purple', xlabel='$M_z$ [t·m]', set_ylabel=False)

    Vy_max = float(np.max(Vy_pos))
    Vy_min = float(np.min(Vy_neg))
    Vz_max = float(np.max(Vz_pos))
    Vz_min = float(np.min(Vz_neg))
    My_max = float(np.max(My_pos))
    My_min = float(np.min(My_neg))
    Mz_max = float(np.max(Mz_pos))
    Mz_min = float(np.min(Mz_neg))
    footer = (f"L={L:.3f}  Vy(+/-)=({Vy_max:.3f}, {Vy_min:.3f}) t  "
              f"Vz(+/-)=({Vz_max:.3f}, {Vz_min:.3f}) t  "
              f"My(+/-)=({My_max:.3f}, {My_min:.3f}) t·m  "
              f"Mz(+/-)=({Mz_max:.3f}, {Mz_min:.3f}) t·m")
    return fig, footer


def page_column_design(ele, tag_txt):
    checks = column_design_checks[ele]
    summary = checks['summary']
    design = column_design_by_element[ele]
    phiPn0_ton = summary['phiPn0']
    Pu_levels = summary.get('Pu_levels', [0.0])
    Pu_levels_plot = [lvl for lvl in Pu_levels if lvl <= phiPn0_ton + 1e-6]
    if len(Pu_levels_plot) == 0:
        Pu_levels_plot = [0.0]
    colors_levels = plt.cm.Greys(np.linspace(0.35, 0.85, len(Pu_levels_plot)))

    def plot_interaction(ax, axis_key, title):
        curve = summary['interaction_plot'][axis_key]
        moment_key = 'My' if axis_key == 'y' else 'Mz'
        ratio_key = 'ratio_y' if axis_key == 'y' else 'ratio_z'
        xlabel = 'My [t-m]' if axis_key == 'y' else 'Mz [t-m]'
        ax.set_title(title, fontsize=11, weight='bold', pad=10)
        ax.plot(curve['Mn'], curve['Pn'], color='tab:blue', lw=2.0, label='Pn-Mn (ϕ=1.0)')
        ax.plot(-curve['Mn'], curve['Pn'], color='tab:blue', lw=2.0)
        ax.plot(curve['phiMn'], curve['phiPn'], color='black', lw=2.0, label='ϕPn-ϕMn')
        ax.plot(-curve['phiMn'], curve['phiPn'], color='black', lw=2.0)
        ax.axhline(phiPn0_ton, color='0.4', lw=0.8, ls='--', label=f'ϕPn0={phiPn0_ton:.1f} t')
        ax.axhline(0.0, color='0.7', lw=0.8, ls=':')

        # La meseta de compresión sigue disponible en los datos de resumen por si se
        # requiere más adelante, pero se omite su trazo horizontal para mantener la
        # gráfica despejada según lo solicitado.

        demand_moments = []
        demand_axial = []
        for combo in combo_names:
            for end_key in ('i', 'j'):
                data = checks[end_key][combo]
                Pu = data['Pu']
                M_val = data[moment_key]
                ratio_val = data[ratio_key]
                ok_point = ratio_val <= 1.0 + 1e-6
                # Colores a tu gusto:
                cap_line_color   = 'gray'           # línea radial (ej. negro)
                cap_marker_color = 'blue'  # “x” de capacidad
                demand_color     = 'tab:green' if ok_point else 'tab:red'  # demanda por estado

                phi_point = data['phi_point_y'] if axis_key == 'y' else data['phi_point_z']
                if phi_point is not None:
                    Pu_cap, M_cap = phi_point
                    if np.isfinite(Pu_cap) and np.isfinite(M_cap):
                        ax.plot([0.0, M_cap], [0.0, Pu_cap], color=cap_line_color,
                                lw=0.7, alpha=0.65, zorder=2)
                        ax.scatter(M_cap, Pu_cap, color=cap_marker_color, marker='x', s=12, zorder=5)
                ax.scatter(M_val, Pu, color=demand_color, marker='o', edgecolor='none', s=8, zorder=4)
                demand_moments.append(abs(M_val))
                demand_axial.append(Pu)

        moment_span = max(demand_moments + curve['Mn'].tolist() + curve['phiMn'].tolist() + [1.0])
        axial_values = demand_axial + curve['Pn'].tolist() + curve['phiPn'].tolist() + [phiPn0_ton, 0.0]
        axial_max = max(axial_values) if axial_values else 0.0
        axial_min = min(axial_values) if axial_values else 0.0
        margin_m = 0.15 * moment_span if moment_span > 1e-6 else 1.0
        margin_p = 0.05 * max(abs(axial_max), abs(axial_min), 1.0)
        ax.set_xlim(-moment_span - margin_m, moment_span + margin_m)
        ax.set_ylim(axial_min - margin_p, axial_max + margin_p)

        #for idx, Pu in enumerate(Pu_levels_plot):
            #if abs(Pu) <= max(abs(axial_max), abs(axial_min)) + 5 * margin_p:
                #ax.axhline(Pu, color=colors_levels[idx], ls=':', lw=0.8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Pu [t]')
        ax.grid(True, ls=':', alpha=0.4)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

    # Hoja 1: propiedades, sección y tabla de combinaciones
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    fig.suptitle('Diseño de Columnas', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
    subheader = (
        f"Columna {tag_txt} · Metodología seccional con refuerzo longitudinal optimizado"
    )
    fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_SUBHEADER_Y,
             subheader, ha='center', va='center', fontsize=11)
    fig.subplots_adjust(top=PAGE_SUBHEADER_Y - 0.055)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.36, 0.2, 0.44], hspace=0.16)

    ax_section = fig.add_subplot(gs[0, 0])
    plot_column_section(ax_section, summary['side'], summary['side'], clear_cover_col,
                        stirrup_diam, design['layout'], summary['rebar_summary'])
    ax_section.set_anchor('C')
    ax_section.set_title('Sección transversal de la columna', fontsize=11, weight='bold', pad=10)
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')
    ax_info.set_anchor('N')
    info_lines = [
        f"Ubicación: nivel {summary['nivel_bottom']} → {summary['nivel_top']} | z={summary['z_bottom']:.2f}→{summary['z_top']:.2f} m",
        f"fc'={fc_col_kgf_cm2:.1f} kg/cm², fy={fy_col_kgf_cm2:.0f} kg/cm², Es={Es_col_kgf_cm2:,.0f} kg/cm²",
        f"Sección {summary['side']:.2f}×{summary['side']:.2f} m  Recubrimiento={clear_cover_col*1000:.0f} mm",
        f"Refuerzo: {design['rebar_summary']} (As={summary['As_total']*1e4:.2f} cm², ρ={summary['rho_long']*100:.2f}%)",
        f"Grupo de niveles: {column_group_label(*summary['level_group']).capitalize()}",
        f"ϕPn0={phiPn0_ton:.2f} t  Pu,max={summary['Pu_max_comp']:.2f} t  Pu,min={summary['Pu_min_ten']:.2f} t",
        f"ratio1={summary['max_ratio_y']:.2f}  ratio2={summary['max_ratio_z']:.2f}",
        f"Índice max Bresler={summary['max_ratio']:.2f} ({'OK' if summary['ok'] else 'No cumple'})"
    ]
    info_lines.append(COLUMN_STEEL_NOTE_SHORT)
    ax_info.text(0.5, 0.96, "\n".join(info_lines), ha='center', va='top', fontsize=9.6,
                transform=ax_info.transAxes, wrap=True)

    ax_table = fig.add_subplot(gs[2, 0])
    ax_table.axis('off')
    table_cols = ['Sección', 'Combo', 'Pu [t]', 'My [t-m]', 'Mz [t-m]', 'ratio1', 'ratio2', 'Índice Bresler']
    table_rows = []
    for end_key, end_label in [('i', 'Base'), ('j', 'Cabeza')]:
        for combo in combo_names:
            data = checks[end_key][combo]
            table_rows.append([
                end_label,
                combo,
                f"{data['Pu']:.2f}",
                f"{data['My']:.2f}",
                f"{data['Mz']:.2f}",
                f"{data['ratio_y']:.2f}",
                f"{data['ratio_z']:.2f}",
                f"{data['ratio']:.2f}"
            ])
    col_widths = [0.1, 0.18, 0.11, 0.11, 0.11, 0.09, 0.09, 0.12]
    table = ax_table.table(cellText=table_rows, colLabels=table_cols,
                           colWidths=col_widths, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(0.98, 1.08)
    for r, row in enumerate(table_rows):
        ratio_vals = [float(row[-3]), float(row[-2]), float(row[-1])]
        if any(val > 1.0 + 1e-3 for val in ratio_vals):
            for c in range(len(table_cols)):
                table[(r+1, c)].set_facecolor('#f8d7da')
    ax_table.text(0.5, -TABLE_CAPTION_GAP,
                  table_caption(('column_combos', ele)),
                  transform=ax_table.transAxes, ha='center', va='top', fontsize=9)
    fig1 = fig

    # Hoja 2: diagramas de interacción apilados
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    fig.suptitle('Diseño de Columnas', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
    fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_SUBHEADER_Y,
             f"Columna {tag_txt} · Diagramas de interacción biaxial",
             ha='center', va='center', fontsize=11)
    fig.subplots_adjust(top=PAGE_SUBHEADER_Y - 0.06)
    gs_inter = fig.add_gridspec(2, 1, height_ratios=[0.5, 0.5], hspace=0.32)
    ax_inter_y = fig.add_subplot(gs_inter[0, 0])
    plot_interaction(ax_inter_y, 'y', 'Interacción My - Pu')
    ax_inter_y.text(0.5, -FIGURE_CAPTION_GAP,
                    figure_caption(('column_interaction_my', ele)),
                    transform=ax_inter_y.transAxes, ha='center', va='top', fontsize=9)
    ax_inter_z = fig.add_subplot(gs_inter[1, 0])
    plot_interaction(ax_inter_z, 'z', 'Interacción Mz - Pu')
    ax_inter_z.text(0.5, -FIGURE_CAPTION_GAP,
                    figure_caption(('column_interaction_mz', ele)),
                    transform=ax_inter_z.transAxes, ha='center', va='top', fontsize=9)
    fig2 = fig

    return fig1, fig2


# ------------------ CONSTRUCCIÓN DEL PDF A4 ------------------
prepare_caption_registry()
reset_page_counter()


def _append_plan(title, subtitle='', level=1):
    page_outline_plan.append({'title': title, 'subtitle': subtitle, 'level': level})


_append_plan('Portada', 'Datos generales del proyecto', level=1)
_append_plan('Vistas Generales', 'Perfiles ortogonales y vista 3D', level=1)
for iz in range(1, nz+1):
    subtitle = f'nivel {iz} · z={Z[iz-1]:.2f}→{Z[iz]:.2f} m'
    _append_plan(f'Planta Nivel {iz}', subtitle, level=2)
_append_plan('Espectro Modal', 'Espectro E.030 y modos dominantes', level=1)
for (ele, iz, iy, ix) in beamsX:
    tag = f'nivel {iz}, y={iy}, vano x={ix}'
    _append_plan(f'Viga-X ele {ele}', tag, level=2)
for (ele, iz, ix, iy) in beamsY:
    tag = f'nivel {iz}, x={ix}, vano y={iy}'
    _append_plan(f'Viga-Y ele {ele}', tag, level=2)
if beam_summary_rows:
    _append_plan('Diseño de Vigas - Resumen', 'Flexión unidireccional por elemento', level=1)
for idx in range(0, len(ordered_beams), 2):
    batch = ordered_beams[idx:idx+2]
    tags = [tag for (_, _, tag) in batch]
    subtitle = ' · '.join(tags)
    _append_plan('Diseño de Vigas - Detalle', subtitle, level=2)
for (ele, ix, iy, iz) in columns_by_level:
    tag = f'(x={ix}, y={iy}, piso {iz}, ele={ele})'
    _append_plan(f'Columna ele {ele} - Envolventes', tag, level=2)
if column_summary_rows:
    _append_plan('Diseño de Columnas - Resumen', 'Demandas Bresler y estado', level=1)
for (ele, ix, iy, iz) in columns_by_level:
    tag = f'(x={ix}, y={iy}, piso {iz}, ele={ele})'
    _append_plan(f'Columna ele {ele} - Propiedades', tag, level=2)
    _append_plan(f'Columna ele {ele} - Diagramas', tag, level=2)

_append_plan('Derivas de Entrepiso', 'Distorsiones máximas en direcciones X e Y', level=1)


plan_iter = iter(page_outline_plan)

with PdfPages(PDF_NAME) as pdf:
    # Portada
    page_info = next(plan_iter)
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_HEADER_Y,
             "REPORTE DE PÓRTICO 3D", ha='center', fontsize=20, weight='bold')
    gamma_conc_t = gamma_conc * kN_TO_TON
    q_losa_t = q_losa * kN_TO_TON
    q_acab_t = q_acab * kN_TO_TON
    q_tabiq_t = q_tabiq * kN_TO_TON
    qv_t = [val * kN_TO_TON for val in qv]
    span_x_desc = ", ".join(f"{val:.2f} m" for val in Lx)
    span_y_desc = ", ".join(f"{val:.2f} m" for val in Ly)
    level_desc = ", ".join(f"{val:.2f} m" for val in H)
    qv_desc = ", ".join(f"{val:.3f} t/m²" for val in qv_t)

    info_header = f"Fecha: {datetime.now():%Y-%m-%d %H:%M}"

    paragraph_1 = (
        f"El pórtico 3D paramétrico (OpenSeesPy) se compone de {nx} vanos en X (∑Lx={sum(Lx):.2f} m; tramos {span_x_desc}) y "
        f"{ny} vanos en Y (∑Ly={sum(Ly):.2f} m; tramos {span_y_desc}), distribuidos en {nz} niveles con alturas {level_desc}. "
        f"Las columnas rectangulares cuentan con {COLUMN_SECTION_SUMMARY} y las vigas con secciones agrupadas por pisos: {beam_section_summary_text} "
        f"(ancho tributario en planta={beam_plan_width:.2f} m). El predimensionamiento inicial adoptó vigas {beam_predim_base:.2f}×{beam_predim_height:.2f} m y columnas {column_predim_side:.2f}×{column_predim_side:.2f} m, "
        f"incrementando dimensiones solo cuando fue necesario para cumplir. Los apoyos base se modelan como {base_fix} (restricciones aplicadas en la base)."
    )

    paragraph_2 = (
        f"La losa de espesor t_losa={t_losa:.3f} m y peso específico γ={gamma_conc_t:.3f} t/m³ produce q_losa={q_losa_t:.3f} t/m². "
        f"A ello se añaden acabados q_acab={q_acab_t:.3f} t/m², tabiquería q_tabiq={q_tabiq_t:.3f} t/m² y cargas vivas reducidas q_viva=({qv_desc}). "
        f"Se consideran los niveles cargados {loaded_levels} para el análisis gravitacional y sísmico."
    )

    paragraph_3 = (
        f"El peso sísmico equivalente asciende a {total_weight_ton:.2f} t (masa={total_mass_ton:.2f} t) con ψ_live={psi_live:.2f}. "
        f"Los parámetros del espectro E.030 son Z={Z_sismo:.2f}, U={U_importancia:.2f}, S={S_suelo:.2f}, R={R_respuesta:.2f}, Tp={Tp:.2f} s y Tl={Tl:.2f} s. "
        f"Los modos fundamentales resultan T₁x={Tx1:.3f} s y T₁y={Ty1:.3f} s, con participaciones de masa de {mode_mass_x_pct:.1f}% y {mode_mass_y_pct:.1f}% (acumulado {mass_ratio_total_X_pct:.1f}% y {mass_ratio_total_Y_pct:.1f}%). "
        f"Los cortantes basales son Vbx={base_shear_x_ton:.2f} t y Vby={base_shear_y_ton:.2f} t, mientras que las derivas máximas de entrepiso alcanzan {drift_x_pct:.2f}% en X y {drift_y_pct:.2f}% en Y con desplazamientos de cubierta Δtecho-X={roof_disp_x_m:.4f} m y Δtecho-Y={roof_disp_y_m:.4f} m."
    )

    wrapped_paragraphs = [textwrap.fill(p, width=110) for p in (paragraph_1, paragraph_2, paragraph_3)]
    info = "\n\n".join([info_header] + wrapped_paragraphs)
    fig.text(PAGE_LEFT, PAGE_TOP - 0.06, info, ha='left', va='top', fontsize=10)
    ax3d = fig.add_subplot(2,1,2, projection='3d')
    ax3d.set_title('Vista 3D (general)', fontsize=12)
    draw_frame_3d(ax3d)
    finalize_page(pdf, fig, page_info)

    # Vistas globales (perfiles y 3D con cargas)
    page_info = next(plan_iter)
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    gs  = fig.add_gridspec(2,2, hspace=0.35, wspace=0.25)
    # Perfil XZ
    axXZ = fig.add_subplot(gs[0,0])
    axXZ.set_title('Perfil-X (proyección X–Z)', fontsize=11, weight='bold')
    for ix in range(nx+1):
        for iz in range(1, nz+1):
            axXZ.plot([X[ix], X[ix]], [Z[iz-1], Z[iz]], color='k', lw=2)
    for iz in range(1, nz+1):
        for ix in range(nx):
            axXZ.plot([X[ix], X[ix+1]], [Z[iz], Z[iz]], color='0.3', lw=2)
    axXZ.set_xlim(-0.5, X[-1]+0.5); axXZ.set_ylim(-0.2, Z[-1]+0.8)
    axXZ.set_xlabel('X [m]'); axXZ.set_ylabel('Z [m]'); axXZ.grid(True, ls=':', alpha=0.35)
    # Perfil YZ
    axYZ = fig.add_subplot(gs[0,1])
    axYZ.set_title('Perfil-Y (proyección Y–Z)', fontsize=11, weight='bold')
    for iy in range(ny+1):
        for iz in range(1, nz+1):
            axYZ.plot([Y[iy], Y[iy]], [Z[iz-1], Z[iz]], color='k', lw=2)
    for iz in range(1, nz+1):
        for iy in range(ny):
            axYZ.plot([Y[iy], Y[iy+1]], [Z[iz], Z[iz]], color='0.3', lw=2)
    axYZ.set_xlim(-0.5, Y[-1]+0.5); axYZ.set_ylim(-0.2, Z[-1]+0.8)
    axYZ.set_xlabel('Y [m]'); axYZ.set_ylabel('Z [m]'); axYZ.grid(True, ls=':', alpha=0.35)
    # 3D general
    ax3d1 = fig.add_subplot(gs[1,0], projection='3d')
    ax3d1.set_title('3D', fontsize=11, weight='bold')
    draw_frame_3d(ax3d1)
    # 3D con CARGAS aplicadas (flechas proporcionales a w)
    ax3d2 = fig.add_subplot(gs[1,1], projection='3d')
    draw_frame_3d_with_loads(ax3d2)
    finalize_page(pdf, fig, page_info)

    # Plantas por nivel (una pagina por nivel)
    for iz in range(1, nz+1):
        page_info = next(plan_iter)
        fig = plt.figure(figsize=(8.27, 11.69))
        apply_page_margins(fig)
        axP = fig.add_subplot(1,1,1)
        draw_plan_level(axP, iz)
        fig.suptitle(f'Planta - Nivel {iz}', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
        finalize_page(pdf, fig, page_info)

    # Espectro modal (antes de diagramas de elementos)
    page_info = next(plan_iter)
    fig, footer = page_spectrum(modal_X, modal_Y)
    finalize_page(pdf, fig, page_info, footer_left=footer)

    # Todas las VIGAS // X
    for (ele, iz, iy, ix) in beamsX:
        page_info = next(plan_iter)
        tag = f"Viga-X (nivel {iz}, y={iy}, vano x={ix}, ele={ele})"
        fig, footer = page_beam(ele, tag, beam_envelopes[ele])
        finalize_page(pdf, fig, page_info, footer_left=footer)

    # Todas las VIGAS // Y
    for (ele, iz, ix, iy) in beamsY:
        page_info = next(plan_iter)
        tag = f"Viga-Y (nivel {iz}, x={ix}, vano y={iy}, ele={ele})"
        fig, footer = page_beam(ele, tag, beam_envelopes[ele])
        finalize_page(pdf, fig, page_info, footer_left=footer)

    # Resumen de diseño de vigas
    if beam_summary_rows:
        page_info = next(plan_iter)
        fig = plt.figure(figsize=(8.27, 11.69))
        apply_page_margins(fig)
        fig.suptitle('Diseño de Vigas', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
        fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_SUBHEADER_Y,
                 'Resumen general de diseño flexional', ha='center', va='center', fontsize=11)
        fig.subplots_adjust(top=PAGE_SUBHEADER_Y - 0.06)
        gs_beam = fig.add_gridspec(2, 1, height_ratios=[0.35, 0.65], hspace=0.05)
        ax_text = fig.add_subplot(gs_beam[0, 0])
        ax_text.axis('off')
        beam_summary_lines = [
            "Metodología: flexión unidireccional con ϕ=0.90, refuerzo mínimo según ACI 318-19.",
            BEAM_STEEL_NOTE,
            f"Secciones adoptadas por niveles: {beam_section_summary_text}"
        ]
        ax_text.text(0.0, 1.0,
                     "\n".join(beam_summary_lines),
                     ha='left', va='top', fontsize=10)
        ax_table = fig.add_subplot(gs_beam[1, 0])
        ax_table.axis('off')
        table_cols = ['Ele', 'Orientación', 'Posición', 'Grupo', 'Sección [m]',
                      'Mu+ [t-m]', 'Mu- [t-m]', 'Ref. inf', 'Ref. sup', 'Demanda max', 'Estado']
        table = ax_table.table(cellText=beam_summary_rows, colLabels=table_cols, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(0.96, 1.10)
        for r, row in enumerate(beam_summary_rows):
            if row[-1] != 'OK':
                for c in range(len(table_cols)):
                    table[(r+1, c)].set_facecolor('#f8d7da')
        if beam_summary_rows:
            ax_table.text(0.5, -TABLE_CAPTION_GAP,
                          table_caption(('beam_summary', 'general')),
                          transform=ax_table.transAxes, ha='center', va='top', fontsize=9)
        finalize_page(pdf, fig, page_info)

    # Paginas detalladas de diseño de vigas (ordenadas por demanda)
    for idx in range(0, len(ordered_beams), 2):
        page_info = next(plan_iter)
        batch = ordered_beams[idx:idx+2]
        entries = [(ele, tag) for (_, ele, tag) in batch]
        fig = build_beam_design_page(entries)
        if fig is not None:
            finalize_page(pdf, fig, page_info)

    # Todas las COLUMNAS
    for (ele, ix, iy, iz) in columns_by_level:
        page_info = next(plan_iter)
        tag = f"(x={ix}, y={iy}, piso {iz}, ele={ele})"
        fig, footer = page_column(ele, tag, column_envelopes[ele])
        finalize_page(pdf, fig, page_info, footer_left=footer)

    # Resumen de diseño de columnas (Bresler)
    if column_summary_rows:
        page_info = next(plan_iter)
        fig = plt.figure(figsize=(8.27, 11.69))
        apply_page_margins(fig)
        fig.suptitle('Diseño de Columnas', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
        fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_SUBHEADER_Y,
                 'Metodología Bresler (ACI 318-19) para columnas rectangulares',
                 ha='center', va='center', fontsize=11)
        fig.subplots_adjust(top=PAGE_SUBHEADER_Y - 0.06)
        summary_text = [
            "Metodología: aproximación de Bresler basada en ACI 318-19 (ecuación 22.5.1.2).",
            f"Secciones rectangulares por grupo: {COLUMN_SECTION_SUMMARY}.",
            COLUMN_STEEL_NOTE_SHORT,
            COLUMN_STEEL_NOTE,
            f"Materiales: fc'={fc_col_kgf_cm2:.1f} kg/cm², fy={fy_col_kgf_cm2:.0f} kg/cm²; factor phi_axial={phi_axial_col:.2f}.",
            f"Capacidades axiales factorizadas: {COLUMN_PHI_SUMMARY}."
        ]
        gs_summary = fig.add_gridspec(2, 1, height_ratios=[0.35, 0.65], hspace=0.05)
        ax_text = fig.add_subplot(gs_summary[0, 0])
        ax_text.axis('off')
        ax_text.text(0.0, 1.0, "\n".join(summary_text), ha='left', va='top', fontsize=10)
        ax_table = fig.add_subplot(gs_summary[1, 0])
        ax_table.axis('off')
        table_cols = ['Ele', '(ix,iy)', 'Nivel', 'Pu_max [t]', 'Pu_min [t]', 'Índice max', 'Estado']
        table = ax_table.table(cellText=column_summary_rows, colLabels=table_cols, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(0.96, 1.12)
        for r, row in enumerate(column_summary_rows):
            if row[-1] != 'OK':
                for c in range(len(table_cols)):
                    table[(r+1, c)].set_facecolor('#f8d7da')
        if column_summary_rows:
            ax_table.text(0.5, -TABLE_CAPTION_GAP,
                          table_caption(('column_summary', 'general')),
                          transform=ax_table.transAxes, ha='center', va='top', fontsize=9)
        finalize_page(pdf, fig, page_info)

    # Paginas detalladas por columna con diagramas biaxiales
    for (ele, ix, iy, iz) in columns_by_level:
        tag = f"(x={ix}, y={iy}, piso {iz}, ele={ele})"
        fig1, fig2 = page_column_design(ele, tag)
        page_info = next(plan_iter)
        finalize_page(pdf, fig1, page_info)
        page_info = next(plan_iter)
        finalize_page(pdf, fig2, page_info)

    page_info = next(plan_iter)
    fig = plt.figure(figsize=(8.27, 11.69))
    apply_page_margins(fig)
    fig.suptitle('Derivas de entrepiso', fontsize=14, weight='bold', y=PAGE_HEADER_Y)
    fig.text((PAGE_LEFT + PAGE_RIGHT) / 2, PAGE_SUBHEADER_Y,
             'Distorsiones máximas relativas por nivel',
             ha='center', va='center', fontsize=11)
    fig.subplots_adjust(top=PAGE_SUBHEADER_Y - 0.08)
    ax = fig.add_subplot(1, 1, 1)
    floors = drift_profile['floors']
    drift_x_levels = drift_profile['ratio_x'] * 100
    drift_y_levels = drift_profile['ratio_y'] * 100
    if floors.size > 0:
        floors_plot = np.insert(floors, 0, 0)
        drift_x_plot = np.insert(drift_x_levels, 0, 0.0)
        drift_y_plot = np.insert(drift_y_levels, 0, 0.0)
        ax.set_yticks(floors)
        ax.set_ylim(0.0, floors_plot.max() + 0.5)
        ax.set_xlabel('Distorsión de entrepiso [%]')
        ax.set_ylabel('Piso')
        ax.grid(True, axis='both', ls=':', alpha=0.5)
    if floors.size > 0:
        ax.plot(drift_x_plot, floors_plot, marker='o', color='tab:orange', lw=2.2, label='Dirección X')
        ax.plot(drift_y_plot, floors_plot, marker='s', color='tab:blue', lw=2.2, label='Dirección Y')
        max_val = max(np.max(np.abs(drift_x_levels)), np.max(np.abs(drift_y_levels)), 0.0)
        if max_val <= 0.0:
            max_val = 0.1
        ax.set_xlim(0.0, max_val * 1.15)
        ax.legend(loc='best')
    else:
        ax.text(0.5, 0.5, 'Sin niveles definidos', ha='center', va='center', transform=ax.transAxes)
    register_figure(('drift_profile', 'xy'), 'Deriva máxima de entrepiso en direcciones X e Y')
    ax.text(0.5, -FIGURE_CAPTION_GAP,
            figure_caption(('drift_profile', 'xy')),
            transform=ax.transAxes, ha='center', va='top', fontsize=9)
    finalize_page(pdf, fig, page_info)

try:
    extra_info = next(plan_iter)
    raise RuntimeError(f'Sobraron entradas en el plan de páginas: {extra_info}')
except StopIteration:
    pass

print(f"PDF generado: {PDF_NAME}")
print(f"Modal X: T1={Tx1:.3f}s, masa mod={mode_mass_x_pct:.1f}%, masa acum={mass_ratio_total_X_pct:.1f}%, Vb={base_shear_x_ton:.2f} t")
print(f"Modal Y: T1={Ty1:.3f}s, masa mod={mode_mass_y_pct:.1f}%, masa acum={mass_ratio_total_Y_pct:.1f}%, Vb={base_shear_y_ton:.2f} t")