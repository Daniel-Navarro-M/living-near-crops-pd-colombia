# Replaces the QGIS shortest-line step: distance (m) from each geocoded life-period residence to the nearest cultivated polygon in CORINE.
# Map year by period: 2005 for 0-17, 18-25, 26-40, 41-60; 2018 for 61+ and the current address. Crop classes follow the exploration notebook's
# 'limpieza': 2.1 transitory crops, 2.2 permanent crops, 2.4.1/2.4.2/2.4.3/2.4.5 crop mosaics; pastures (2.3, 2.4.4) excluded.
import pandas as pd, geopandas as gpd
coord = pd.read_csv('D:/Proyecto_TQ/DatosGeoref20260708/DatosGeoref/coordenadas.csv', sep=';')
periods = {'0-17': ('0', '2005'), '18-25': ('18', '2005'), '26-40': ('26', '2005'), '41-60': ('41', '2005'), '61': ('61', '2018'), 'current': ('Residencia', '2018')}
pts = pd.concat([pd.DataFrame({'Record ID': coord['Record ID'], 'period': p, 'map': yr, 'x': pd.to_numeric(coord[f'{c}_coord x' if c != 'Residencia' else 'Residencia x'], errors='coerce'), 'y': pd.to_numeric(coord[f'{c}_coord y' if c != 'Residencia' else 'Residencia y'], errors='coerce')}) for p, (c, yr) in periods.items()])
print('coordinates total:', len(pts), '| numeric:', pts['x'].notna().sum())
pts = pts[pts['x'].between(-79.1, -66.8) & pts['y'].between(-4.3, 13.5)]; print('inside Colombia bbox (rest = geocoding failures, dropped):', len(pts))
pts = gpd.GeoDataFrame(pts.reset_index(drop=True), geometry=gpd.points_from_xy(pts['x'], pts['y']), crs='EPSG:4326').to_crs('EPSG:3116')
bbox = tuple(pts.to_crs('EPSG:4326').total_bounds + [-0.02, -0.02, 0.02, 0.02])
crop = lambda s: s.str.strip().str.startswith(('2.1', '2.2', '2.4.1', '2.4.2', '2.4.3', '2.4.5'))
pts['dist_m'] = float('nan')
for yr, path, col in [('2005', 'D:/Proyecto_TQ/CORINE_05_09_COL_c.geojson', 'LEYENDA'), ('2018', 'D:/Proyecto_TQ/cobertura_tierra_clc_2018.geojson', 'leyenda')]:
    poly = gpd.read_file(path, bbox=bbox, columns=[col]); poly = poly[crop(poly[col])].to_crs('EPSG:3116'); print(f'CORINE {yr}: cultivated polygons in study bbox:', len(poly))
    sub = pts[pts['map'] == yr]; near = gpd.sjoin_nearest(sub, poly[['geometry']], how='left', max_distance=1000, distance_col='d')
    pts.loc[sub.index, 'dist_m'] = near.groupby(level=0)['d'].min().reindex(sub.index).values
    print(f'  {yr} points: {len(sub)}; with a cultivated polygon within 1 km: {pts.loc[sub.index, "dist_m"].notna().sum()}; median distance {pts.loc[sub.index, "dist_m"].median():.0f} m')
pts[['Record ID', 'period', 'map', 'dist_m']].to_csv('D:/Proyecto_TQ/article/georef_distances.csv', index=False); print('wrote georef_distances.csv', pts.shape)
