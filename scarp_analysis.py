from __future__ import division
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from itertools import accumulate
import pandas as pd


def classify(data):
    # Dispatch W- / E-tilting scarps
    claS = []
    for key, group in groupby(data):
        claS.append(list(group))
    return claS


def indexing(data):
    # Assign index values to each scarp
    flat_D = [item for sublist in data for item in sublist]
    indexes = list(range(len(flat_D)))

    data_N = []
    cnt = 0
    for i in range(len(data)):
        dat_TMP, idx_TMP = [], []
        for j in range(len(data[i])):
            n = j + cnt
            dat_TMP.append(data[i][j])
            idx_TMP.append(indexes[n])

        data_N.append([idx_TMP, dat_TMP])
        cnt += len(data[i])
    return data_N


def regroup(data):
    # Group scarps based on ternary values (0,1,2) in different lists
    # 	1 = West-tilted scarps
    # 	2 = East-tilted scarps
    # 	pass if flat terrain
    s_1, s_2 = [], []
    for i in range(len(data)):
        if 1 in data[i][1]:
            s_1.append(data[i])
        elif 2 in data[i][1]:
            s_2.append(data[i])
        else:
            pass
    return s_1, s_2


def find_nearest(array, value):
    # find nearest value
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def measure_scarp_extent(sca_dst, sca_hgt):
    # Measures the total scarp extension in profile

    # Rule out scarps with 1 indice i.e., of length < 100m in ArcGIS
    sca_dst = [i for i in sca_dst if len(i) != 1]

    tot_lgths = 0  # total length for all scarps
    lgths = []
    for i in range(len(sca_dst)):
        lgth = 0  # length of individual scarp
        for j in range(len(sca_dst[i])):
            if sca_dst[i][j] != sca_dst[i][-1]:
                try:
                    dx = np.abs(sca_dst[i][j + 1] - sca_dst[i][j])
                    dy = np.abs(sca_hgt[i][j + 1] - sca_hgt[i][j])
                    hp = np.sqrt(np.square(dx) + np.square(dy))  # hypotenus
                    lgth = lgth + hp
                    tot_lgths = tot_lgths + hp
                except:
                    pass
            else:
                pass
        if lgth != 0:
            lgths.append(lgth)
        else:
            pass
    return tot_lgths, lgths


def measure_fault_heave(sca_dst, sca_hgt):
    # Measures the total scarp extension on profile

    # Rule out scarps with 1 indice i.e., of length < 100m in ArcGIS
    sca_dst = [i for i in sca_dst if len(i) != 1]
    sca_hgt = [i for i in sca_hgt if len(i) != 1]

    heaves, throws = [], []
    for s in sca_dst:
        heave = np.abs(s[0] - s[-1])
        heaves.append(heave)

    for s in sca_hgt:
        throw = np.abs(s[0] - s[-1])
        throws.append(throw)
    tot_extent = sum(heaves)
    return tot_extent, heaves, throws


def get_coordinates(sca, dist, dpth):
    #
    sca_dst, sca_dpt = [], []
    for i in range(len(sca)):
        sca_dst.append([dist[j] for j in sca[i]])
        sca_dpt.append([dpth[j] for j in sca[i]])
    return sca_dst, sca_dpt


def reshape_scarps(x, y):
    # Smooth scarps based on up/down limit of scarps given depth/distance values
    coord_x, coord_y, thetas_rsh, thetas_avg, thetas_lst = [], [], [], [], []
    for pt_dx, pt_dy in zip(x, y):
        # Do not take scarps with 2 points or less (because irrelevant)
        if len(pt_dx) < 2:
            coord_x.append(pt_dx)
            coord_y.append(pt_dy)
        else:
            x_TMP, y_TMP = [], []  # initialized lists
            # Reverse list order for E-tilted scarps (avoid changing up/dwn indexes)
            inverted = 0
            if pt_dy[0] < pt_dy[-1]:
                pt_dx = pt_dx[::-1]
                pt_dy = pt_dy[::-1]
                inverted = 1

            up_x = pt_dx[0]
            up_y = pt_dy[0]

            # Iterate through sub-lists
            # we look for maximum angle (from horizontal) to which the "real" fault is built on
            angles = []
            for dst, dpt in zip(pt_dx, pt_dy):
                if dst != pt_dx[0]:
                    dy = np.abs(up_y - dpt)
                    dx = np.abs(up_x - dst)
                    rad = dy / dx
                    theta = np.arctan(rad) * 57.2958
                    angles.append(theta)
                else:
                    angles.append(0)  # to keep the list length consistent

            # Angle and index to/of the point from which the linear reshape will be done
            # Notes
            # 		take min angle, i.e. line closer to vertical axis
            theta_n = max(angles)
            idx = np.asarray(angles).argmax()

            thetas_lst.append(angles)
            thetas_rsh.append(theta_n)
            thetas_avg.append(np.mean(angles))

            # Takes upper points
            x_TMP.append(up_x)

            for n in range(1, len(pt_dx)):
                # Only reshape points not forming the regress line
                if (pt_dx[n] != up_x) and (pt_dx[n] != pt_dx[idx]):
                    dy_n = np.abs(up_y - pt_dy[n])
                    dx_n = np.abs(up_x - pt_dx[n])
                    AB = np.sqrt(np.square(dx_n) + np.square(dy_n))  # hypotenus
                    dr_n = dx_n / AB
                    alpha = np.arccos(dr_n) * 57.2958

                    beta = (90 - alpha) - (90 - theta_n)
                    BBp = AB * np.sin(np.radians(beta))

                    sigma = theta_n
                    BCp = BBp / np.sin(np.radians(sigma))

                    # B' coordinates?
                    if inverted == 1:
                        # for east tilted scarps
                        new_x = pt_dx[n] + BCp
                    else:
                        # for west tilted scarps
                        new_x = pt_dx[n] - BCp
                    x_TMP.append(new_x)

                # Takes 2nd points used for the regress line def.
                elif pt_dx[n] == pt_dx[idx]:
                    x_TMP.append(pt_dx[n])
                else:
                    pass

            # If the inversion is True, undo the inversion
            if inverted == 1:
                x_TMP = x_TMP[::-1]
                pt_dy = pt_dy[::-1]

            coord_x.append(x_TMP)
            coord_y.append(pt_dy)

    return coord_x, coord_y, thetas_rsh, thetas_avg, thetas_lst


def evaluate_reshape(origin_x, reshap_x):
    # Evaluate the gap between the non-reshaped / reshaped scarp
    # i.e. the distance from fault corrected (or dispersion of debris during mass movement)
    all_gaps = []

    for s in range(len(origin_x)):
        gaps = []
        for i, j in zip(origin_x[s], reshap_x[s]):
            # Gap (horizontal separation or dx)
            gaps.append(np.abs(i - j))
        all_gaps.append(gaps)
    return all_gaps


def evaluate_reshape_areas(origin_x, reshap_x, origin_y):
    # List of lists of gaps
    all_areas, all_areas_sum = [], []
    #
    for i in range(len(origin_x)):
        areas = []
        for j in range(len(origin_x[i])):
            #
            try:
                x_start = origin_x[i][j]
                x_rsh_start = reshap_x[i][j]
                x_next = origin_x[i][j + 1]
                x_rsh_next = reshap_x[i][j + 1]
                y_start = origin_y[i][j]
                y_next = origin_y[i][j + 1]
            except:
                continue
            # area = ((a+b) * dy )/ 2
            a = np.abs(x_start - x_rsh_start)
            b = np.abs(x_next - x_rsh_next)
            dy = np.abs(y_start - y_next)
            area = ((a + b) * dy) / 2
            areas.append(area)
        all_areas.append(areas)
        all_areas_sum.append(sum(areas))
    return all_areas, all_areas_sum


def gaps_profiling(model, gaps_idx, gaps):
    # Create profile of gaps (horizontal distance original fault - corrected fault)
    gaps_idx_ = [i for s in gaps_idx for i in s]
    gaps_ = [i for s in gaps for i in s]

    new_profile = []
    cnt = 0
    for i in range(len(model)):
        # If the index exist in those of the gaps, add gaps value
        # otherwise, put null value
        if i in gaps_idx_:
            try:
                new_profile.append(gaps_[cnt])
                cnt = cnt + 1
            except:
                new_profile.append(0)
        elif i + len(model) in gaps_idx_:
            try:
                new_profile.append(gaps_[cnt])
                cnt = cnt + 1
            except:
                new_profile.append(0)
        else:
            new_profile.append(0)
    #
    return new_profile


def progression(cnt, data):
    lgth = len(data) / 10
    if cnt == 0:
        print("0% done...")
    elif cnt == int(lgth):
        print("10% done...")
    elif cnt == int(2 * lgth):
        print("20% done...")
    elif cnt == int(3 * lgth):
        print("30% done...")
    elif cnt == int(4 * lgth):
        print("40% done...")
    elif cnt == int(5 * lgth):
        print("50% done...")
    elif cnt == int(6 * lgth):
        print("60% done...")
    elif cnt == int(7 * lgth):
        print("70% done...")
    elif cnt == int(8 * lgth):
        print("80% done...")
    elif cnt == int(9 * lgth):
        print("90% done...")
    elif cnt == int(10 * lgth) - 2:
        print("100% done...")


def normalize(data):
    return np.asarray([((i - np.min(data)) / (np.max(data) - np.min(data))) for i in data])


class quantifier(object):
    """Quantify scarps properties and the importance of reshaping"""
    def __init__(self):
        # initialize containers
        self.W_strain, self.E_strain, self.total_strain = ([] for i in range(3))
        self.gaps_W, self.gaps_E, self.gaps_W_areas, self.gaps_E_areas, \
        self.gaps_W_areas_sum, self.gaps_E_areas_sum = ([] for i in range(6))
        self.angles_rsh_W, self.angles_avg_W, self.angles_rsh_E, self.angles_avg_E = ([] for i in range(4))
        self.lgths_W, self.lgths_E, self.lgths_rsh_W, self.lgths_rsh_E = ([] for i in range(4))
        self.heaves_W, self.heaves_E, self.throws_W, \
        self.throws_E, self.heaves_rsh_W, self.heaves_rsh_E = ([] for i in range(6))
        self.x_mid_coord, self.y_mid_coord, self.dist_E, self.dist_W = ([] for i in range(4))
        self.gaps_volumes_sum = []
        self.gaps_final, self.areas_final, self.depths_, self.fids = ([] for i in range(4))

    def run_analysis(self, data, profiles):
        # Analyze scarps properties in every profile
        print('Working on profiles...')
        cnt = 0
        for pval in profiles:
            progression(cnt=cnt, data=data)
            cnt = cnt + 1

            # Get variables
            depths = data[pval]['Depth'].tolist()
            distances = data[pval]['Distance'].tolist()
            sca_id = data[pval]['Scarps'].tolist()
            x_0 = data[pval]['X_Coord'].tolist()[0]
            x_f = data[pval]['X_Coord'].tolist()[-1]
            y_0 = data[pval]['Y_Coord'].tolist()[0]
            y_f = data[pval]['Y_Coord'].tolist()[-1]
            fid = data[pval]['FID'].tolist()

            # Calculate mid-coordinates of the profile
            # Notes
            # Profile center determines where scarps (W or E) types should be separated in the analysis
            # e.g. the 1st 25km of profile = West-tilting scarps, while remaining 25km = East-tilting ones
            x_mid = (x_0 + x_f) / 2
            y_mid = (y_0 + y_f) / 2
            self.x_mid_coord.append(x_mid)
            self.y_mid_coord.append(y_mid)

            # Classify scarps
            # Notes
            # sca_1: West tilting
            # sca_2: East tilting
            sca_class = classify(data=sca_id)
            sca_index = indexing(data=sca_class)
            sca_1, sca_2 = regroup(data=sca_index)
            sca_1 = [i[0] for i in sca_1]
            sca_2 = [i[0] for i in sca_2]

            # Gaps of profile
            # i.e. horizontal distance between original scarp - corrected scarp at every vertex of the profile
            idexs = [dist / 100 for dist in
                     distances]  # divided by 100m, the original interval between vertex in profiles
            middle_val = find_nearest(array=np.asarray(idexs), value=250)  # find value center of profile
            middle_idx = np.where(np.asarray(idexs) == middle_val)[0][0]  # get index of this center

            # Indexes of inward scarps
            # i.e. scarp facing the profile center
            if sca_1 != []:
                # Get distance and depth of scarps West tilted
                sca_W_dst, sca_W_dpt = get_coordinates(sca=sca_1, dist=distances, dpth=depths)
                # Reshape or get corrected scarp
                sca_W_rsh_dst, sca_W_rsh_dpt, sca_W_rsh_ang, sca_W_avg_ang, sca_W_ang = reshape_scarps(x=sca_W_dst,
                                                                                                       y=sca_W_dpt)
                # Evaluate the reshaped scarp
                # i.e. calculate the gaps between a vertex of corrected_original scarp
                sca_W_gaps = evaluate_reshape(origin_x=sca_W_dst, reshap_x=sca_W_rsh_dst)
                # calculate areas from 4 vertex between original and reshaped scarps
                sca_W_gaps_areas, sca_W_gaps_areas_sum = evaluate_reshape_areas(origin_x=sca_W_dst,
                                                                                reshap_x=sca_W_rsh_dst,
                                                                                origin_y=sca_W_dpt)
                sca_gaps_pfl_1 = gaps_profiling(model=idexs[:middle_idx], gaps_idx=sca_1, gaps=sca_W_gaps)

                self.gaps_W.append(sca_W_gaps)
                self.gaps_W_areas.append(sca_W_gaps_areas)
                self.gaps_W_areas_sum.append(sca_W_gaps_areas_sum)
                self.angles_rsh_W.append(sca_W_rsh_ang)
                self.angles_avg_W.append(sca_W_avg_ang)

                # -------------------------

                # length of W-tilting scarps (1) total profile, and (2) individual scarps
                p_sca_W_tot_lgths, sca_W_lgths = measure_scarp_extent(sca_dst=sca_W_dst, sca_hgt=sca_W_dpt)
                # same but for reshaped scarps
                p_sca_W_rsh_tot_lgths, sca_W_rsh_lgths = measure_scarp_extent(sca_dst=sca_W_rsh_dst, sca_hgt=sca_W_dpt)
                # hypotenuse, heave and throw (original scarps)
                p_sca_W_extension, sca_W_heaves, sca_W_throws = measure_fault_heave(sca_dst=sca_W_dst,
                                                                                    sca_hgt=sca_W_dpt)
                # hypotenuse, heave and throw (reshaped scarps)
                p_sca_W_rsh_extension, sca_W_rsh_heaves, sca_W_rsh_throws = measure_fault_heave(sca_dst=sca_W_rsh_dst,
                                                                                                sca_hgt=sca_W_dpt)
                # calculate fault strain (original scarps)
                p_W_strain = p_sca_W_extension / np.abs(distances[middle_idx] - distances[0])

                self.lgths_W.append(sca_W_lgths)
                self.lgths_rsh_W.append(sca_W_rsh_lgths)
                self.W_strain.append(p_W_strain)
                self.heaves_W.append(sca_W_heaves)
                self.heaves_rsh_W.append(sca_W_rsh_heaves)
                self.throws_W.append(sca_W_throws)
                self.dist_W.append(sca_W_dst)

            else:
                # if sca_1 does not exist
                sca_gaps_pfl_1 = [0 for i in range(len(idexs[:middle_idx]))]        # for ArcGIS update of transects
                p_sca_W_extension = 0.

            if sca_2 != []:
                sca_E_dst, sca_E_dpt = get_coordinates(sca=sca_2, dist=distances, dpth=depths)
                sca_E_rsh_dst, sca_E_rsh_dpt, sca_E_rsh_ang, sca_E_avg_ang, sca_E_ang = reshape_scarps(x=sca_E_dst,
                                                                                                       y=sca_E_dpt)
                sca_E_gaps = evaluate_reshape(origin_x=sca_E_dst, reshap_x=sca_E_rsh_dst)
                sca_E_gaps_areas, sca_E_gaps_areas_sum = evaluate_reshape_areas(origin_x=sca_E_dst,
                                                                                reshap_x=sca_E_rsh_dst,
                                                                                origin_y=sca_E_dpt)
                sca_gaps_pfl_2 = gaps_profiling(model=idexs[middle_idx:], gaps_idx=sca_2, gaps=sca_E_gaps)

                self.gaps_E.append(sca_E_gaps)
                self.gaps_E_areas.append(sca_E_gaps_areas)
                self.gaps_E_areas_sum.append(sca_E_gaps_areas_sum)
                self.angles_rsh_E.append(sca_E_rsh_ang)
                self.angles_avg_E.append(sca_E_avg_ang)

                # -------------------------

                # length of W-tilting scarps (1) total profile, and (2) individual scarps
                p_sca_E_tot_lgths, sca_E_lgths = measure_scarp_extent(sca_dst=sca_E_dst, sca_hgt=sca_E_dpt)
                # same but for reshaped scarps
                p_sca_E_rsh_tot_lgths, sca_E_rsh_lgths = measure_scarp_extent(sca_dst=sca_E_rsh_dst, sca_hgt=sca_E_dpt)
                # hypotenuse, heave and throw (original scarps)
                p_sca_E_extension, sca_E_heaves, sca_E_throws = measure_fault_heave(sca_dst=sca_E_dst,
                                                                                    sca_hgt=sca_E_dpt)
                # hypotenuse, heave and throw (reshaped scarps)
                p_sca_E_rsh_extension, sca_E_rsh_heaves, sca_E_rsh_throws = measure_fault_heave(sca_dst=sca_E_rsh_dst,
                                                                                                sca_hgt=sca_E_dpt)
                # calculate fault strain (original scarps)
                p_E_strain = p_sca_E_extension / np.abs(distances[middle_idx] - distances[-1])

                self.lgths_E.append(sca_E_lgths)
                self.lgths_rsh_E.append(sca_E_rsh_lgths)
                self.E_strain.append(p_E_strain)
                self.heaves_E.append(sca_E_heaves)
                self.heaves_rsh_E.append(sca_E_rsh_heaves)
                self.throws_E.append(sca_E_throws)
                self.dist_E.append(sca_E_dst)
            else:
                sca_gaps_pfl_2 = [0 for i in range(len(idexs[middle_idx:]))]
                p_sca_E_extension = 0.

            # get total strain for the profile considered (original scarps)
            p_total_strain = (p_sca_W_extension + p_sca_E_extension) / (np.abs(distances[-1] - distances[0]))
            self.total_strain.append(p_total_strain)

            # collect vertex data (gaps, gaps_areas) for updating transects in ArcGIS
            sca_gaps_pfl = sca_gaps_pfl_1 + sca_gaps_pfl_2
            self.gaps_final = self.gaps_final + sca_gaps_pfl

            sca_areas_pfl = sca_gaps_pfl_1 + sca_gaps_pfl_2
            self.areas_final = self.areas_final + sca_areas_pfl

            self.fids = self.fids + fid
            self.depths_ = self.depths_ + depths

        # get volumes of debris mass deposited over scarps
        volumes_W = [np.sum(e) for e in self.gaps_W_areas_sum]
        volumes_E = [np.sum(e) for e in self.gaps_E_areas_sum]
        self.gaps_volumes_sum.append(volumes_W + volumes_E)

        # Output to be used for updating transects in ArcGIS
        output = [[self.fids[i], self.depths_[i], self.gaps_final[i], self.areas_final[i]] for i in range(len(self.fids))]
        self.output = np.array(sorted(output, key=lambda x: x[0]))

    def write_outputs(self,out_path):
        inputD = {'FID': self.output[:,0],
                  'Depth': self.output[:,1],
                  'Gaps': self.output[:,2],
                  'Gaps_areas': self.output[:,3]}
        inputC = ['FID', 'Depth', 'Gaps', 'Gaps_areas']
        # DataFrame creation
        frame = pd.DataFrame(data=inputD, columns=inputC)
        # Output destination (for CSV export)
        frame.to_csv(path_or_buf=out_path, index=False, header=True)
        print("Results exported to CSV file.")

    def plot_heaves(self, limit=999999):
        """Plot scarps heaves (original) vs heaves (reshaped), given throw
        :param limit: to limit number of profiles analyzed"""
        # Get heaves and throws of profiles
        v2_flat = [e for s in self.heaves_rsh_W for e in s] + [e for s in self.heaves_rsh_E for e in s]
        v1_flat = [e for s in self.heaves_W for e in s] + [e for s in self.heaves_E for e in s]
        variable = [e for s in self.throws_W for e in s] + [e for s in self.throws_E for e in s]

        # Set figure configuration
        canvas = plt.figure(1)
        rect = canvas.patch
        rect.set_facecolor('white')
        ax = canvas.add_subplot(1, 1, 1)
        ax.yaxis.grid(True)  # horizontal lines
        ax.xaxis.grid(True)  # vertical lines

        # Set color mapping
        jet = plt.get_cmap('jet')  # use '6' for discretization of gradient into 6 sections
        norm = colors.Normalize(vmin=min(variable[:limit]), vmax=max(variable[:limit]))  # normalize (0-1)
        scalarMap = cmx.ScalarMappable(norm=norm, cmap=jet)  # color scaling using norm
        colorVal = scalarMap.to_rgba(variable[:limit])  # generates (r,g,b,a) tuples
        colorVal = colorVal.tolist()
        scalarMap.set_array(variable[:limit])

        # Plot
        _ = plt.scatter(v1_flat[:limit], v2_flat[:limit], color=colorVal, s=3)
        plt.colorbar(scalarMap)
        plt.show()

    def plot_strain(self, transects):
        """Plot tectonic strain
        :param transects: define which transects to account"""
        # Set figure configuration
        canvas = plt.figure(1)
        rect = canvas.patch
        rect.set_facecolor('white')
        ax = canvas.add_subplot(1,1,1)
        ax.yaxis.grid(color='black', linestyle='dotted', linewidth=1) # horizontal lines
        ax.xaxis.grid(color='black', linestyle='dotted', linewidth=1) # vertical lines

        cnt = 0
        for i,j,k,l in zip(self.heaves_W, self.heaves_E, self.dist_W, self.dist_E):
            if cnt in transects:
                try:
                    s = i + j
                    s = [0] + list(accumulate(s))
                    d1 = [np.mean(sc) for sc in k]
                    d2 = [np.mean(sc) for sc in l]
                    d = [0] + d1 + d2
                    ax.plot(d,s, 'black', linewidth=0.3)
                    ax.scatter(d,s, color='black', s=4)
                except:
                    pass
            else:
                pass
            cnt += 1

        ax.set_ylim(bottom=0, top=9500)
        ax.set_xlim(left=0, right=50000)
        ax.set_xticks(np.arange(0, 50000, 10000))
        ax.set_yticks(np.arange(0, 9500, 1000))
        plt.show()

    def plot_histogram(self, var, bin_size=50):
        """Plot histograms"""
        # Flattening
        flat = [e for s in var for e in s if str(e) != 'nan']
        # Configure canvas
        canvas = plt.figure(1)
        rect = canvas.patch
        rect.set_facecolor('white')
        ax = canvas.gca()
        ax.yaxis.grid(color='black', linestyle='dotted', linewidth=1)
        ax.xaxis.grid(color='black', linestyle='dotted', linewidth=1)
        # Get histograms
        n, bins, patches = ax.hist(flat, bin_size, facecolor='blue', alpha=1, edgecolor='black', linewidth=1)
        plt.show()

    def plot_ratios(self, var1, var2, bin_size):
        """Plot histograms for ratios | used to compare original vs reshaped scarps"""
        # Flattening
        flat1 = [e for s in var1 for e in s if str(e) != 'nan']
        flat2 = [e for s in var2 for e in s if str(e) != 'nan']
        # Configure canvas
        canvas = plt.figure(1)
        rect = canvas.patch
        rect.set_facecolor('white')
        ax = canvas.gca()
        ax.yaxis.grid(color='black', linestyle='dotted', linewidth=1)
        ax.xaxis.grid(color='black', linestyle='dotted', linewidth=1)
        # Get histograms
        ratio = [i / j if j != 0 else 0 for i,j in zip(flat1, flat2)]
        n, bins, patches = ax.hist(ratio, bin_size, facecolor='blue', alpha=1, edgecolor='black', linewidth=1)
        plt.show()

    def plot_dualscatter(self, varx, vary):
        """Compare graphically 2 variables by scatter plot"""
        # Set figure configuration
        canvas = plt.figure(1)
        rect = canvas.patch
        rect.set_facecolor('white')
        ax = canvas.add_subplot(1, 1, 1)
        ax.yaxis.grid(True)  # horizontal lines
        ax.xaxis.grid(True)  # vertical lines

        flat1 = varx[0] + varx[1]
        flat2 = [sum(i) for i in vary[0]] + [sum(i) for i in vary[1]]
        ls1, ls2 = zip(*sorted(zip(flat1, flat2)))
        plt.scatter(ls1, ls2, s=10, facecolors='none', edgecolor='black')
        plt.show()