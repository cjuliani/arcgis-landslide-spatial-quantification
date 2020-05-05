# Mass potential quantification of debris on scarps via geometric reshaping and statistical analysis 

This demo was developed to estimate mass movements generated by normal fault scarps on either sides of a slow-spreading mid-ocean ridge using bathymetric data. First, transects (or profiles) are drawn in ArcGIS perpendicularly to ridges, then depth and position of the vertex from each transect (100-meters interval) are considered  for geometric reshaping of scarps - vertex identify a west-tilting or east-tilting scarp (1 or 2 respectively) or not (0), see data file. 
Secondly, a data processing pipeline (scarp_analysis.py) correct the shape of scarps from the  upper part of a fault surface given that the slip surface in the subseafloor supposedly follow a straight plan of sliding (see figure below). The topographic correction for a given vertex is used to calculate the horizontal distance  between the corrected fault line and the potential debris materials shaping the original scarp landform (i.e., the “uncorrected” topography) - this is the "gap". This distance outlines the ridge-ward extent of a potential debris mass on or in close proximity to the corrected fault segment. The thickness of this mass can be approximated by summing the areas  delimited between two successive topographic vertice. As the fault reshaping affects the measurable angle of fault emergence, the angular measure (and other properties such as the fault heave or throw) for corrected and non-corrected fault segments are calculated.
Finally, vertex of transects are updated so that a vertex spatially located in GIS consists of an estimate of the gap or mass area generated on the scarp.

| ![alt text](https://raw.githubusercontent.com/cjuliani/arcgis-landslide-spatial-quantification/master/geometric_reshaping.PNG) |
|:--:|
| *Cross-sectional view of a geometrically corrected fault offset affected by debris mass. The corrected fault line is drawn from the reference points p1 and p2; the former is a vertex of the topography below which slope is the steepest uphill, while the latter is obtained after seeking the lowest angle (θ) formed between the vertical axis and the segment joining p1 and any of the topographic vertices. A vertex to be corrected is situated at an angle β from the vertical axis. An estimated area of deposition (Si) is calculated longitudinally between two vertices of the original topography, and their corrected coordinates. Summing all areas Si gives a total deposit area, which can be multiplied by 1-m of lateral extent to estimate the volume of displaced debris. Horizontal distance between vertices is 100 m.* 

| ![alt text](https://raw.githubusercontent.com/cjuliani/arcgis-landslide-spatial-quantification/master/throw_heave_statistics.PNG) |
|:--:|
| *Examples of frequency distributions of fault throw, heave, angles and length ratio (uncorrected/reshaped) for a population of ridge-ward facing fault scarps.* 

| ![alt text](https://raw.githubusercontent.com/cjuliani/arcgis-landslide-spatial-quantification/master/tectonic_extension.PNG) |
|:--:|
| *Cumulative apparent heave profiles for transects sampled throughout the ridge. Tectonic strain is indicated in percentage. Note the flat-ramp configuration below 10% strain where flats represent areas lacking fault scarp. * 

| ![alt text](https://raw.githubusercontent.com/cjuliani/arcgis-landslide-spatial-quantification/master/case_study.png) |
|:--:|
| *(a) Shaded relief (top) and slope map (bottom) of the north-western rift valley margin. Dashed and dotted white lines represent horseshoe-shaped scars traces (Si) marking slope collapses, heads of landslides (Hi) and toes of mass deposits (Ti). Deposits of wasted materials from landslides (Li) are either sustained upstream (L1), discharged downslope (L2) or deposited downstream (L3). Closer to the neo-volcanic zone, young scarp may have potential deposits (marked by “?”) but these remain difficult to discriminate given that volcanism is rather prevalent at these areas. Nevertheless, they outline similar U-shaped deposition patterns distinguished on young fault scarps. (b) Results of predicted mass wasting evidences for the case study (see Fig.2). Dispersion of mass deposits is calculated from the horizontal distance between the corrected fault line and the fault scarp topography for every transect vertices. A kernel density is then applied for smoothing purpose. * 
