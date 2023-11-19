import pandas as pd
import numpy as np
import os, sys
from monai.data import CacheDataset
from monai.transforms import Randomizable



removed = '/home/yaoqy/Breast_DFS/data/datalists/processed_data_lists/20220907/removed.csv'
removed_id = pd.read_csv(removed,dtype={'ID':str,'isremoved':bool})
removed_id = removed_id[removed_id['isremoved']].values[:,0]
print(len(removed_id))


class DFSDataset(Randomizable, CacheDataset):
    def __init__(
            self,
            csv_path = '',
            data_df = None,
            transform = None,
            phase = 'train',
            input_sequence = ['T1C','T2WI'],
            seed = 0,
            cache_num = sys.maxsize,
            cache_rate: float = 1.0,
            num_workers: int = 0,
            roi_dir = '',
        ) -> None:       
        assert not (len(csv_path)==0 and (data_df is None))
        if data_df is not None:
            self.data_df = data_df
        else:
            self.data_df = pd.read_csv(csv_path,dtype={'ID':str,'dyn_fix':str})
        self.set_random_state(seed=seed)
        self.phase = phase
        self.input_sequence = input_sequence
        self.roi_dir = roi_dir

        self.data_df = self.data_df[~self.data_df['ID'].isin(removed_id)] 
        self.data_df = self.data_df.dropna(how='any',subset=input_sequence+['dfs_status','dfs_time','ID'])

        # self.indices: np.ndarray = np.array([])
        data = self._generate_data_list()
        self.sample_data = data[0:12]
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )
        
    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.
        """
        return self.indices

    def randomize(self, data) -> None:
        self.R.shuffle(data)

    def _generate_data_list(self,):
        datalist = []
        for i,row in self.data_df.iterrows():
            ID = row['ID']
            tmp_data = {'dfs_ID':np.str(ID),'dfs_status':row['dfs_status'],'dfs_time':row['dfs_time'],'is_dyn':os.path.basename(row['T1C'])}

            tmp_data['roi_path'] = f"{self.roi_dir}/{ID}_roi.pkl"
            if not os.path.exists(tmp_data['roi_path']):
                continue 
            if self.phase=='train' and row['dfs_status']==1:
                for _ in range(5): #5
                    datalist.append(tmp_data)
            else:
                datalist.append(tmp_data)    
        return datalist
