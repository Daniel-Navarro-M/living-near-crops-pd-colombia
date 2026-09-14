# DANE MGN 2024 class of the current residence: point-in-polygon query of each geocoded residence against the 'Clase' layer
# (1. CABECERA, 2. CENTRO POBLADO, 3. RURAL DISPERSO) of the DANE map service. Rural in the analysis = any class other than cabecera.
# Residences with no coordinate inside Colombia, or falling in no polygon (2 points), are missing.
import pandas as pd, requests, time
B = 'https://portalgis.dane.gov.co/mparcgis/rest/services/MGN2024/Serv_CapasMGN_2024/MapServer/324/query'
coord = pd.read_csv('D:/Proyecto_TQ/DatosGeoref20260708/DatosGeoref/coordenadas.csv', sep=';')
pts = pd.DataFrame({'Record ID': coord['Record ID'], 'x': pd.to_numeric(coord['Residencia x'], errors='coerce'), 'y': pd.to_numeric(coord['Residencia y'], errors='coerce')})
pts = pts[pts['x'].between(-79.1, -66.8) & pts['y'].between(-4.3, 13.5)]; print('current residences geocoded inside Colombia:', len(pts), 'of', len(coord))
out = []
for _, r in pts.iterrows():
    for attempt in range(3):
        try:
            f = requests.get(B, params=dict(geometry=f'{r.x},{r.y}', geometryType='esriGeometryPoint', inSR=4326, spatialRel='esriSpatialRelIntersects', outFields='mpio_cdpmp,Clase', returnGeometry='false', f='json'), timeout=40).json().get('features', [])
            out.append((r['Record ID'], f[0]['attributes']['Clase'] if f else None, f[0]['attributes']['mpio_cdpmp'] if f else None)); break
        except Exception:
            time.sleep(2)
    else: out.append((r['Record ID'], None, None))
d = pd.DataFrame(out, columns=['Record ID', 'dane_clase', 'dane_mpio']); d.to_csv('D:/Proyecto_TQ/article/dane_clase_residencia.csv', index=False)
print(d['dane_clase'].value_counts(dropna=False).to_dict(), '| wrote dane_clase_residencia.csv')
