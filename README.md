# Submarine mass potential quantification from seabed surface model

The method estimates mass movements generated by tectonic features along transects of fault escarpments. The fault plane is first estimated given the vertex coordinates using data processing pipeline (scarp_analysis.py); hypothetically, the slip surface supposedly follows a straight plan of sliding (see figure below). Then, one calculate the horizontal distance between the fault plane and the potential debris mass shaping the original escarpment. This distance outlines the ridge-ward extent of debris mass whose thickness is approximated by summing areas delimited between two successive vertices of the fault plan and original landform. Here, the estimated fault plan depends on the angle of fault emergence.

| ![alt text](https://raw.githubusercontent.com/cjuliani/arcgis-landslide-spatial-quantification/master/geometric_reshaping.PNG) |
|:--:|
| *Cross-sectional view of a geometrically estimated fault plan underlying a debris mass. The estimate considers the reference points p1 and p2; the former is a vertex below which slope is the steepest uphill, while the latter is obtained after seeking the lowest angle (θ) formed between the vertical axis and the segment joining p1 and any of the topographic vertices. A vertex to be corrected is situated at an angle β from the vertical axis. An estimated area of deposition (Si) is calculated longitudinally between two vertices of the original landform, and their corrected coordinates. Summing all areas Si gives a total deposit area, which can be multiplied by 1-m of lateral extent to estimate the volume of displaced debris. Horizontal distance between vertices is 100 m.* 

| ![alt text](https://raw.githubusercontent.com/cjuliani/arcgis-landslide-spatial-quantification/master/case_study.png) |
|:--:|
| *(a) Shaded relief (top) and slope map (bottom) of the north-western rift valley margin. Dashed and dotted white lines represent horseshoe-shaped scars traces (Si) marking slope collapses, heads of landslides (Hi) and toes of mass deposits (Ti). Deposits of wasted materials from landslides (Li) are either sustained upstream (L1), discharged downslope (L2) or deposited downstream (L3). Closer to the neo-volcanic zone, young scarp may have potential deposits (marked by “?”) but these remain difficult to discriminate given that volcanism is rather prevalent at these areas. Nevertheless, they outline similar U-shaped deposition patterns distinguished on young fault scarps. (b) Results of predicted mass wasting evidences for the case study (see Fig.2). Dispersion of mass deposits is calculated from the horizontal distance between the corrected fault line and the fault scarp topography for every transect vertices. A kernel density is then applied for smoothing purpose.* 

| ![alt text](https://raw.githubusercontent.com/cjuliani/arcgis-landslide-spatial-quantification/master/throw_heave_statistics.PNG) |
|:--:|
| *Examples of frequency distributions of fault throw, heave, angles and length ratio for a population of fault plans.* 

| ![alt text](https://raw.githubusercontent.com/cjuliani/arcgis-landslide-spatial-quantification/master/tectonic_extension_.PNG) |
|:--:|
| *Cumulative apparent heave profiles for transects sampled throughout the ridge. Tectonic strain is indicated in percentage. Note the flat-ramp configuration below 10% strain where flats represent areas lacking fault scarp.* 
