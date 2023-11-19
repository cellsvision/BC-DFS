import pandas as pd
import numpy as np
import os, sys
from monai.data import CacheDataset
from monai.transforms import (
    Randomizable,
)


class SegDataset(Randomizable, CacheDataset):
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
            # skip_id=[],
            # return_df = True,
        ) -> None:       
        assert not (len(csv_path)==0 and (data_df is None))
        if data_df is not None:
            self.data_df = data_df
        else:
            self.data_df = pd.read_csv(csv_path,dtype={'ID':str,'dyn_fix':str})
        self.set_random_state(seed=seed)
        self.phase = phase
        self.input_sequence = input_sequence

        self.data_df = self.data_df.dropna(how='any',subset=input_sequence)

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
            tmp_data = {'seg_ID':np.str(ID),'t1c_range':[-1,-1]}
            # tmp_data = {}

            t1c_path = row['T1C']
            if ('T1C*' in t1c_path) or ('T1C_' in t1c_path):
                if isinstance(row['dyn_fix'],str):
                    tmp_data['T1C'] = os.path.dirname(row['T1C'])+f"/T1C_{int(float(row['dyn_fix']))}"
                else:
                    tmp_data['T1C'] = os.path.dirname(row['T1C'])+f"/T1C_2"
            else:
                tmp_data['T1C'] = t1c_path
            tmp_data['T2WI'] = row['T2WI']
            tmp_data['label'] = row['segmentation']
            datalist.append(tmp_data)
        return datalist