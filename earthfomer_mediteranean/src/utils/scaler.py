
import numpy as np

class DataScaler:
    def __init__(self, config) -> None:
        self.mode = config['mode']
    
    def normalize(self,data, min, max):
        return (data - min) / (max - min)
    
    def standardize(self,data, mean, std):
        return (data - mean) / std

    def remove_outliers_23std(self, data, clipped_min, clipped_max):
        data = np.clip(data.data[0, :].astype(float), clipped_min, clipped_max)
        bottom = clipped_max - clipped_min
        bottom[bottom == 0] = "nan"
        data = (data - clipped_min) / bottom
        return np.nan_to_num(data, 0)

    def scale(self, data, stats):
        mean,std,min,max = stats.values()
        if self.mode == "normalize":
            return self.normalize(data, min, max)
        if self.mode == "std23":
            return self.remove_outliers_23std(data, min, max)
        elif self.mode == "standardize":
            return self.standardize(data, mean, std)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
