import os as os


def crs2wsg84(gdf_fnc):
    """
    Function to convert a geodataframe to wsg84
    """
    wgs84_crs = gdf_fnc.crs.to_string().split(" +up")[0]
    gdf_fnc = gdf_fnc.to_crs(wgs84_crs)
    return gdf_fnc

# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# ----------------------------------------------------------------------
