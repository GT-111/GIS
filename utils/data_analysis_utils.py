
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, parafac
import matplotlib.pyplot as plt

def plot_factor(arr, cfg, title):
    rank = cfg.tensor_decomposition.rank
    weights, factors = non_negative_parafac(tl.tensor(arr), rank=rank, n_iter_max=200)
    num_factors = len(factors)
    factor_day, factor_interval, factor_wp, factor_type = factors
    
    # Set the size of the overall figure
    fig = plt.figure(figsize=(24, 16))
    
    # Create subplots with proper spacing
    gs = fig.add_gridspec(rank, num_factors, hspace=0.15, wspace=0.2, width_ratios=[5, 3, 3, 3])
    
    for i in range(rank):
        for j, factor in enumerate([factor_day, factor_interval, factor_wp, factor_type]):
            ax = fig.add_subplot(gs[i, j])
            
            if j == 0:
                # day factor
                ax.plot(factor[:, i])
                if i == 0:
                    ax.set_title(f'Date')
            elif j == 1:
                # interval factor
                time_interval = cfg.tensor_decomposition.time_interval
                # generate x-axis
                time_intervals = []
                for time_idx in range(int(24 // time_interval)):
                    time_intervals.append(f'{time_idx * time_interval}H - {(time_idx + 1) * time_interval}H')
                
                ax.bar(time_intervals, factor[:, i])
                if i == 0:
                    ax.set_title(f'Time')
            elif j == 2:
                # waypoint factor
                ax.bar(np.arange(len(factor[:, i])), factor[:, i])
                if i == 0:
                    ax.set_title(f'Waypoint')
            elif j == 3:
                # type factor
                vessel_types = cfg.tensor_decomposition.vessel_types
                ax.bar(vessel_types, factor[:, i])
                if i == 0:
                    ax.set_title(f'Type')
            ax.yaxis.set_visible(False)
    plt.savefig(f'./{title}.png')