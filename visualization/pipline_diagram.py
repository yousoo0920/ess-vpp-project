# pipline_diagram.py
"""
Organic 3D Pipeline (branches / fusion / feedback), Plotly 3D
- 축 숫자/제목/그리드/배경 완전히 숨김 (깨끗한 뷰)
- Annotation: 머리는 노드 좌표(x,y,z) 고정, 라벨은 ax/ay(픽셀)로만 이동 → 일관된 방향/거리
- Curved (Bezier) edges + cones for direction
- Layered cuboids: Data / Feature Engineering / Model / EMS / Logging / Retraining
"""

import math
from typing import Dict, Tuple, List, Optional
import numpy as np
import plotly.graph_objects as go

HTML_OUT = "pipeline_3d.html"

# =============================
# 1) Sections (3D cuboids)
# =============================
SECTIONS = [
    {"name": "데이터 소스",         "center": (-1.6,  0.2,  0.70), "size": (1.6, 1.2, 0.12), "color": "#93c5fd"},
    {"name": "특징공학(입력 벡터)",   "center": (-0.3,  0.1,  0.30), "size": (1.6, 1.0, 0.12), "color": "#86efac"},
    {"name": "모델 코어",           "center": ( 0.9,  0.1, -0.10), "size": (1.6, 0.9, 0.14), "color": "#fde68a"},
    {"name": "EMS·스케줄러",         "center": ( 1.5, -0.6, -0.45), "size": (1.0, 0.8, 0.12), "color": "#fbcfe8"},
    {"name": "로깅·모니터링",         "center": ( 0.4, -0.8, -0.65), "size": (1.4, 0.7, 0.12), "color": "#a5b4fc"},
    {"name": "재학습(피드백 루프)",   "center": (-0.9, -0.7, -0.30), "size": (1.3, 0.7, 0.12), "color": "#fca5a5"},
]

def cuboid(center, size, color, name, opacity=0.16):
    """정석 큐보이드(12 삼각형) + 상단 라벨"""
    cx, cy, cz = center
    w, h, d = size
    x = [cx - w/2, cx + w/2]
    y = [cy - h/2, cy + h/2]
    z = [cz - d/2, cz + d/2]
    V = [
        (x[0], y[0], z[0]), (x[1], y[0], z[0]), (x[1], y[1], z[0]), (x[0], y[1], z[0]),
        (x[0], y[0], z[1]), (x[1], y[0], z[1]), (x[1], y[1], z[1]), (x[0], y[1], z[1]),
    ]
    vx, vy, vz = zip(*V)
    i = [0,1,2, 0,2,3,  4,6,5, 4,7,6,  0,4,5, 0,5,1,  1,5,6, 1,6,2,  2,6,7, 2,7,3,  3,7,4, 3,4,0]
    j = [1,2,3, 3,0,1,  6,5,4, 7,6,4,  4,5,1, 1,0,4,  5,6,2, 2,1,5,  6,7,3, 3,2,6,  7,4,0, 4,3,7]
    k = [2,3,0, 1,1,2,  5,4,6, 6,5,7,  5,1,0, 4,4,0,  6,2,1, 5,2,1,  7,3,2, 6,6,3,  4,0,3, 7,7,0]
    mesh = go.Mesh3d(x=vx, y=vy, z=vz, i=i, j=j, k=k,
                     opacity=opacity, color=color, name=name,
                     flatshading=True, hovertext=name, hoverinfo="text")
    title = go.Scatter3d(x=[cx], y=[cy], z=[cz + d/2 + 0.03],
                         mode="text", text=[f"<b>{name}</b>"], showlegend=False)
    return [mesh, title]

# =============================
# 2) Nodes (spread out to avoid overlap)
# =============================
nodes: Dict[str, Dict] = {
    # 데이터 소스
    "KMA 단기예보 API":     {"pos": (-2.2,  0.7,  0.74), "color": "#3b82f6", "size": 8},
    "KPX 수요예측 API":     {"pos": (-2.4, -0.1,  0.74), "color": "#3b82f6", "size": 8},
    "발전량(태양광/풍력)":   {"pos": (-1.7, -0.6,  0.74), "color": "#3b82f6", "size": 8},
    "운영제약/공지":        {"pos": (-1.2,  0.5,  0.74), "color": "#3b82f6", "size": 8},

    # 특징공학
    "정합/결측처리":         {"pos": (-0.9,  0.5,  0.34), "color": "#16a34a", "size": 9},
    "스케일링":             {"pos": (-0.9,  0.1,  0.34), "color": "#16a34a", "size": 9},
    "파생특징(증감률 등)":     {"pos": (-0.6, -0.3,  0.34), "color": "#16a34a", "size": 9},
    "전일_출력제한량":       {"pos": (-0.3,  0.3,  0.34), "color": "#22c55e", "size": 8},
    "출력제한_변화율":       {"pos": (-0.3,  0.0,  0.34), "color": "#22c55e", "size": 8},
    "전일_태양광":          {"pos": ( 0.0,  0.3,  0.34), "color": "#22c55e", "size": 8},
    "전일_풍력":            {"pos": ( 0.0,  0.0,  0.34), "color": "#22c55e", "size": 8},
    "출력비율":             {"pos": ( 0.0, -0.3,  0.34), "color": "#22c55e", "size": 8},
    "Feature Tensor":       {"pos": ( 0.3,  0.1,  0.30), "color": "#10b981", "size": 10},

    # 모델 코어
    "LSTM-1":               {"pos": ( 0.7,  0.25, -0.06), "color": "#f59e0b", "size": 10},
    "LSTM-2":               {"pos": ( 0.7, -0.05, -0.06), "color": "#f59e0b", "size": 10},
    "Conv1D-분기":           {"pos": ( 0.7, -0.35, -0.06), "color": "#f59e0b", "size": 10},
    "Attention":            {"pos": ( 1.05,  0.10, -0.10), "color": "#fb923c", "size": 10},
    "Ensemble Fusion":      {"pos": ( 1.30,  0.05, -0.14), "color": "#f97316", "size": 10},
    "Post-Processing":      {"pos": ( 1.55,  0.00, -0.18), "color": "#f97316", "size": 10},

    # EMS·스케줄러
    "ESS 스케줄러":          {"pos": ( 1.95, -0.40, -0.46), "color": "#ec4899", "size": 10},
    "제약기반 의사결정":      {"pos": ( 1.65, -0.75, -0.46), "color": "#ec4899", "size": 10},

    # 로깅·모니터링
    "결과 로그(csv)":         {"pos": ( 0.7, -0.85, -0.68), "color": "#6366f1", "size": 9},
    "대시보드":              {"pos": ( 0.3, -0.95, -0.68), "color": "#4f46e5", "size": 9},
    "알람/이상탐지":          {"pos": ( 1.1, -0.95, -0.68), "color": "#4f46e5", "size": 9},

    # 재학습 루프
    "데이터 버전관리":         {"pos": (-0.4, -0.65, -0.32), "color": "#ef4444", "size": 9},
    "오프라인 재학습":        {"pos": (-0.8, -0.75, -0.32), "color": "#ef4444", "size": 9},
    "가중치 배포":           {"pos": (-1.2, -0.65, -0.32), "color": "#ef4444", "size": 9},
}

# =============================
# 3) Organic Edges (curves, bi-directional flags)
# =============================
Edge = Tuple[str, str, str, Optional[Tuple[float,float,float]], bool]
edges: List[Edge] = [
    # 데이터 소스 → 특징공학(분기)
    ("KMA 단기예보 API", "정합/결측처리", "curve", (0.0, 0.2, 0.1), False),
    ("KPX 수요예측 API", "정합/결측처리", "curve", (0.0,-0.2, 0.1), False),
    ("발전량(태양광/풍력)", "정합/결측처리", "curve", (0.2,-0.1, 0.08), False),
    ("운영제약/공지", "정합/결측처리", "curve", (-0.2,0.1, 0.08), False),

    # 특징공학 내부 흐름
    ("정합/결측처리", "스케일링", "curve", (0.0, -0.15, 0.0), False),
    ("정합/결측처리", "파생특징(증감률 등)", "curve", (0.1, -0.1, 0.0), False),
    ("스케일링", "전일_출력제한량", "curve", (0.1, 0.1, 0.0), False),
    ("스케일링", "출력제한_변화율", "curve", (0.05, -0.05, 0.0), False),
    ("파생특징(증감률 등)", "전일_태양광", "curve", (0.1, 0.15, 0.0), False),
    ("파생특징(증감률 등)", "전일_풍력", "curve", (0.1, 0.05, 0.0), False),
    ("파생특징(증감률 등)", "출력비율", "curve", (0.1,-0.05, 0.0), False),

    # 집계 → Feature Tensor
    ("전일_출력제한량", "Feature Tensor", "curve", (0.1, 0.05, -0.05), False),
    ("출력제한_변화율", "Feature Tensor", "curve", (0.1, 0.02, -0.05), False),
    ("전일_태양광", "Feature Tensor", "curve", (0.1,-0.02, -0.05), False),
    ("전일_풍력", "Feature Tensor", "curve", (0.1,-0.05, -0.05), False),
    ("출력비율", "Feature Tensor", "curve", (0.1,-0.08, -0.05), False),

    # 모델 다중 경로 + 집계
    ("Feature Tensor", "LSTM-1", "curve", (0.2, 0.10, -0.20), False),
    ("Feature Tensor", "LSTM-2", "curve", (0.2, -0.05, -0.20), False),
    ("Feature Tensor", "Conv1D-분기", "curve", (0.2, -0.18, -0.20), False),

    ("LSTM-1", "Attention", "curve", (0.2, -0.02, -0.08), False),
    ("LSTM-2", "Attention", "curve", (0.2, 0.00, -0.08), False),
    ("Conv1D-분기", "Attention", "curve", (0.2, 0.04, -0.08), False),

    ("Attention", "Ensemble Fusion", "curve", (0.15, 0.00, -0.06), False),
    ("Ensemble Fusion", "Post-Processing", "curve", (0.18, 0.00, -0.06), False),

    # 모델 → EMS (양방향 케이스 포함)
    ("Post-Processing", "ESS 스케줄러", "curve", (0.25, -0.25, -0.20), True),
    ("Post-Processing", "제약기반 의사결정", "curve", (0.10, -0.35, -0.15), False),

    # EMS → 로깅/모니터링
    ("ESS 스케줄러", "결과 로그(csv)", "curve", (-0.8, -0.25, -0.10), False),
    ("제약기반 의사결정", "대시보드", "curve", (-0.5, -0.25, -0.08), True),

    # 모니터링 상호작용
    ("결과 로그(csv)", "대시보드", "curve", (-0.2, -0.05, 0.0), False),
    ("대시보드", "알람/이상탐지", "curve", (0.4, 0.02, 0.0), False),
    ("알람/이상탐지", "ESS 스케줄러", "curve", (0.6, 0.2, 0.0), True),

    # 재학습 루프
    ("결과 로그(csv)", "데이터 버전관리", "curve", (-0.4, 0.25, 0.12), False),
    ("데이터 버전관리", "오프라인 재학습", "curve", (-0.3, -0.15, -0.02), False),
    ("오프라인 재학습", "가중치 배포", "curve", (-0.25, 0.20, 0.00), False),
    ("가중치 배포", "LSTM-1", "curve", (-0.1, 0.45, 0.25), False),
    ("가중치 배포", "LSTM-2", "curve", (-0.1, 0.30, 0.25), False),
    ("가중치 배포", "Conv1D-분기", "curve", (-0.1, 0.15, 0.25), False),
]

# =============================
# 4) Bezier utilities
# =============================
def p_add(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def bezier_points(p0, p1, ctrl=(0,0,0), steps=30):
    c = p_add(((p0[0]+p1[0])/2, (p0[1]+p1[1])/2, (p0[2]+p1[2])/2), ctrl)
    t = np.linspace(0, 1, steps)
    pts = []
    for tt in t:
        a = (1-tt)**2
        b = 2*(1-tt)*tt
        c2 = tt**2
        x = a*p0[0] + b*c[0] + c2*p1[0]
        y = a*p0[1] + b*c[1] + c2*p1[1]
        z = a*p0[2] + b*c[2] + c2*p1[2]
        pts.append((x,y,z))
    return pts

def tangent_at(pts, frac=0.70):
    idx = max(1, min(len(pts)-2, int(len(pts)*frac)))
    x0,y0,z0 = pts[idx-1]
    x1,y1,z1 = pts[idx+1]
    v = (x1-x0, y1-y0, z1-z0)
    n = math.sqrt(v[0]**2+v[1]**2+v[2]**2) or 1.0
    return (pts[idx], (v[0]/n, v[1]/n, v[2]/n))

# =============================
# 5) Build curves + cones
# =============================
edge_lines_x, edge_lines_y, edge_lines_z = [], [], []
cone_x, cone_y, cone_z, cone_u, cone_v, cone_w = [], [], [], [], [], []

def add_curve(a, b, ctrl=(0,0,0)):
    p0 = nodes[a]["pos"]; p1 = nodes[b]["pos"]
    pts = bezier_points(p0, p1, ctrl=ctrl, steps=30)
    xs, ys, zs = zip(*pts)
    edge_lines_x.extend(list(xs) + [None])
    edge_lines_y.extend(list(ys) + [None])
    edge_lines_z.extend(list(zs) + [None])

    (mx,my,mz), (ux,uy,uz) = tangent_at(pts, frac=0.70)
    L = 0.16
    cone_x.append(mx); cone_y.append(my); cone_z.append(mz)
    cone_u.append(ux*L); cone_v.append(uy*L); cone_w.append(uz*L)

for a,b,kind,ctrl,bi in edges:
    if kind == "curve":
        add_curve(a, b, ctrl or (0,0,0))
        if bi:
            rc = ctrl or (0,0,0)
            add_curve(b, a, (-rc[0]*0.8, rc[1]*0.8, -rc[2]*0.8))

edge_trace = go.Scatter3d(
    x=edge_lines_x, y=edge_lines_y, z=edge_lines_z,
    mode="lines", line=dict(width=4), hoverinfo="none", name="흐름(곡선)"
)
arrow_cones = go.Cone(
    x=cone_x, y=cone_y, z=cone_z, u=cone_u, v=cone_v, w=cone_w,
    sizemode="absolute", sizeref=0.18, anchor="tail", showscale=False, name="방향"
)

# =============================
# 6) Nodes + Annotations (head at node, label via ax/ay)
# =============================
node_dots = go.Scatter3d(
    x=[nodes[n]["pos"][0] for n in nodes],
    y=[nodes[n]["pos"][1] for n in nodes],
    z=[nodes[n]["pos"][2] for n in nodes],
    mode="markers",
    marker=dict(size=[nodes[n]["size"] for n in nodes],
                color=[nodes[n]["color"] for n in nodes],
                line=dict(width=1)),
    text=[n for n in nodes], hoverinfo="text", name="노드"
)

# 라벨 방향/거리(픽셀) — 요청 노드(KPX/KMA) 정밀 튜닝 포함
ARROW_PIX = {
    "KPX 수요예측 API":  {"ax": -110, "ay":  40},
    "KMA 단기예보 API":  {"ax": -110, "ay": -40},

    "발전량(태양광/풍력)": {"ax":  100, "ay": -30},
    "운영제약/공지":     {"ax":  110, "ay":  30},

    "정합/결측처리":      {"ax": -120, "ay":  40},
    "스케일링":          {"ax": -120, "ay": -40},
    "파생특징(증감률 등)":  {"ax":  120, "ay": -50},
    "전일_출력제한량":     {"ax": -120, "ay":  30},
    "출력제한_변화율":     {"ax": -120, "ay": -30},
    "전일_태양광":        {"ax":  110, "ay":  40},
    "전일_풍력":          {"ax":  110, "ay": -40},
    "출력비율":           {"ax":  100, "ay": -50},
    "Feature Tensor":     {"ax":  110, "ay":  20},

    "LSTM-1":            {"ax":  110, "ay":  50},
    "LSTM-2":            {"ax":  110, "ay": -50},
    "Conv1D-분기":        {"ax":  100, "ay": -50},
    "Attention":         {"ax":  110, "ay":  40},
    "Ensemble Fusion":   {"ax":  110, "ay":  30},
    "Post-Processing":   {"ax":  110, "ay":  20},

    "ESS 스케줄러":       {"ax":  120, "ay":  30},
    "제약기반 의사결정":     {"ax":  110, "ay": -60},

    "결과 로그(csv)":      {"ax": -100, "ay":  40},
    "대시보드":           {"ax": -110, "ay":  40},
    "알람/이상탐지":       {"ax":  110, "ay":  40},

    "데이터 버전관리":      {"ax": -110, "ay":  50},
    "오프라인 재학습":     {"ax": -110, "ay": -60},
    "가중치 배포":        {"ax": -120, "ay":  40},
}

def annotation_for(name, default_ax=80, default_ay=0, box_shift=(0,0), standoff=6):
    x, y, z = nodes[name]["pos"]
    p = ARROW_PIX.get(name, {"ax": default_ax, "ay": default_ay})
    xshift, yshift = box_shift
    return dict(
        x=x, y=y, z=z,
        text=f"<b>{name}</b>",
        showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.2,
        ax=p["ax"], ay=p["ay"], standoff=standoff,
        xanchor="left", yanchor="middle",
        xshift=xshift, yshift=yshift,
        bordercolor="#111", borderwidth=1, borderpad=4,
        bgcolor="rgba(255,255,255,0.92)", opacity=0.98,
        font=dict(size=12)
    )

annotations = [annotation_for(n) for n in nodes.keys()]

# =============================
# 7) Figure & Layout (축 숫자/제목 제거)
# =============================
fig = go.Figure()
for s in SECTIONS:
    fig.add_traces(cuboid(s["center"], s["size"], s["color"], s["name"], opacity=0.16))

fig.add_trace(edge_trace)
fig.add_trace(arrow_cones)
fig.add_trace(node_dots)

fig.update_layout(
    title="<b>Curtailment Prediction · Organic 3D Pipeline (Branches · Fusion · Feedback)</b>",
    scene=dict(
        xaxis=dict(showbackground=False, zeroline=False, showgrid=False,
                   showticklabels=False, title=""),
        yaxis=dict(showbackground=False, zeroline=False, showgrid=False,
                   showticklabels=False, title=""),
        zaxis=dict(showbackground=False, zeroline=False, showgrid=False,
                   showticklabels=False, title=""),
        aspectmode="data"
    ),
    margin=dict(l=0, r=0, t=60, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    scene_annotations=annotations
)

fig.write_html(HTML_OUT, include_plotlyjs="inline", full_html=True, auto_open=True)
print(f"[OK] 3D 인터랙티브 파일 저장 및 오픈: {HTML_OUT}")
