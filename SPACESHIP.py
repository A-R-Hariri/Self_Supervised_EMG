import os, h5py
from os.path import join
import numpy as np, pandas as pd
import re

from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler


# ======== CSV → H5 ========

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def process_user(
    user_path,
    out_path,
    subject_id):

    rep = 0
    
    with h5py.File(out_path, "w") as h5:
        column_names = ['time']
        column_names.extend([f"ch{i}" for i in range(1, 9)])
        while True:
            rep += 1
            try:
                data = pd.read_csv(join(user_path, f"emg{rep}.csv") , delimiter=' ', header=None, names=column_names)
            except Exception as e:
                break
                

            rep_grp = h5.create_group(f"rep_{rep}")

            emg = np.concatenate([np.expand_dims(data[f'ch{ch}'].to_numpy(), -1) 
                                    for ch in range(1,9)], -1).astype(np.float32)

            rep_grp.create_dataset("emg", data=emg)
            rep_grp.create_dataset("subject", data=subject_id)
            rep_grp.create_dataset("rep", data=rep)
            rep_grp.create_dataset("gesture", data=99)

        print(f"Finished subject={subject_id} | "
                f"reps={rep-2} | "
                f"out={out_path}")


# ======== DATASET WALKER ========

def process_dataset(root_in, root_out):
    os.makedirs(root_out, exist_ok=True)
    path = join(root_in, 'data')
    user_dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    user_dirs.sort(key=natural_sort_key)
    user_dirs

    for subject_id, user_dir in enumerate(user_dirs):
        in_path = join(path, user_dir)
        out_path = os.path.join(
            root_out, f"{user_dir}.h5")

        process_user(
            user_path=in_path,
            out_path=out_path,
            subject_id=subject_id)


# ======== LIBEMG DATASET ========

class SPACESHIP(Dataset):
    def __init__(self, dataset_folder="spaceship"):
        super().__init__(
            sampling={"": 0},
            num_channels={"": 0},
            recording_device=[""],
            num_subjects=16,
            gestures="",
            num_reps="",
            description="",
            citation="",
        )
        self.dataset_folder = dataset_folder


    def _get_odh(
        self,
        processed_root,
        subjects,
        channel_last,
    ):
        odh = OfflineDataHandler()

        odh.subjects = []
        odh.classes = []
        odh.reps = []
        odh.extra_attributes = ["subjects", "classes", "reps"]

        for file in sorted(os.listdir(processed_root)):
            with h5py.File(join(processed_root, file), "r") as file:

                for r in file:
                    f = file[r]
                    subject = int(f["subject"][()])
                    if subjects is not None and subject not in subjects:
                        continue

                    gst = int(f["gesture"][()])
                    rep_id = int(f["rep"][()])
                    emg = f["emg"][:]
                    emg = emg.astype(np.float32)

                    if not len(emg):
                        continue

                    if not channel_last:
                        emg = emg.T

                    odh.data.append(emg)
                    odh.classes.append(np.ones((len(emg), 1), 
                                                dtype=np.int64) * gst)
                    odh.subjects.append(np.ones((len(emg), 1), 
                                                dtype=np.int64) * subject)
                    odh.reps.append(np.ones((len(emg), 1), 
                                            dtype=np.int64) * rep_id)
        return odh


    def prepare_data(
        self,
        channel_last=True,
        subjects=None):
        processed = self.dataset_folder + "_PROCESSED"

        if not os.path.exists(processed):
            process_dataset(
                self.dataset_folder, processed)

        odh = self._get_odh(
            processed,
            subjects,
            channel_last)

        return odh
