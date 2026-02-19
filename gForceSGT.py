import os
import h5py
import numpy as np
import pandas as pd
import re
from os.path import join

from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler


REP_FORMS = {
    'steadystate': 0,
    'limbpositions': 1,
    }


# ======== UTILS  ========

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


# ======== PREPROCESSING  ========

def process_user(subject_path, out_path, subject_id):
    # Allowed sub-directories
    target_folders = ['steadystate', 'limbpositions']
    
    # Regex to parse C (Class/Gesture) and R (Repetition)
    # Pattern: C_0_R_0_emg.csv
    filename_pattern = re.compile(r"C_(\d+)_R_(\d+)_emg\.csv")

    unique_counter = 0

    with h5py.File(out_path, "w") as h5:
        for folder in target_folders:
            folder_path = join(subject_path, folder)
            
            if not os.path.exists(folder_path):
                continue

            files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
            files.sort(key=natural_sort_key)

            for f_name in files:
                match = filename_pattern.search(f_name)
                if not match:
                    continue

                gesture_id = int(match.group(1))
                rep_id = int(match.group(2))

                # Read CSV: 8 columns, no header
                try:
                    full_file_path = join(folder_path, f_name)
                    data = pd.read_csv(full_file_path, header=None, delimiter=',')
                    
                    # Ensure strictly 8 columns
                    if data.shape[1] != 8:
                        # Fallback for potential whitespace delimiters if standard csv fails
                        data = pd.read_csv(full_file_path, header=None, delim_whitespace=True)
                        if data.shape[1] != 8:
                            print(f"Skipping {f_name}: Invalid column count {data.shape[1]}")
                            continue

                    emg_matrix = data.values.astype(np.float32)

                    # Create unique HDF5 group (cannot use rep_id as key due to duplicates across folders)
                    grp_name = f"segment_{unique_counter}"
                    unique_counter += 1
                    
                    rep_grp = h5.create_group(grp_name)

                    rep_grp.create_dataset("emg", data=emg_matrix)
                    rep_grp.create_dataset("subject", data=subject_id)
                    rep_grp.create_dataset("rep", data=rep_id)
                    rep_grp.create_dataset("gesture", data=gesture_id)
                    rep_grp.create_dataset("rep_form", data=REP_FORMS[folder])

                except Exception as e:
                    print(f"Error processing {f_name}: {e}")

    print(f"Subject {subject_id} processed. Output: {out_path}")


def process_dataset(root_in, root_out):
    os.makedirs(root_out, exist_ok=True)
    
    # Structure defined as data_sgt/Day 1/S#
    base_path = join(root_in, "Day 1")
    
    if not os.path.exists(base_path):
        # Fallback if "Day 1" is not present or root_in already includes it
        base_path = root_in

    user_dirs = [f for f in os.listdir(base_path) if os.path.isdir(join(base_path, f)) and f.startswith('S')]
    user_dirs.sort(key=natural_sort_key)

    for user_dir in user_dirs:
        # Extract ID from 'S0', 'S1' -> 0, 1
        try:
            subject_id = int(user_dir.replace('S', ''))
        except ValueError:
            continue

        in_path = join(base_path, user_dir)
        out_path = join(root_out, f"{user_dir}.h5")

        process_user(
            subject_path=in_path,
            out_path=out_path,
            subject_id=subject_id)


# ======== DATASET CLASS ========

class gForceSGT(Dataset):
    def __init__(self, dataset_folder="data_sgt"):
        # TODO: Configure metadata
        # Assuming 6 subjects so far
        super().__init__(
            sampling={"": 0}, # Fill if Fs is known, else 0
            num_channels={"": 8},
            recording_device=["Generic 8ch"],
            num_subjects=6,
            gestures="",
            num_reps="",
            description="SGT Data - Limb Positions & Steady State",
            citation="",
        )
        self.dataset_folder = dataset_folder

    def _get_odh(self, processed_root, subjects, channel_last):
        odh = OfflineDataHandler()
        odh.subjects = []
        odh.classes = []
        odh.reps = []
        odh.rep_forms = []
        odh.extra_attributes = ["subjects", "classes", 
                                "reps", "rep_forms"]

        # Load all h5 files in directory
        files = sorted([f for f in os.listdir(processed_root) if f.endswith('.h5')], key=natural_sort_key)

        for file_name in files:
            file_path = join(processed_root, file_name)
            
            with h5py.File(file_path, "r") as hf:
                # Iterate over all segments in the file
                for seg_key in hf.keys():
                    group = hf[seg_key]
                    
                    subject = int(group["subject"][()])
                    
                    # Subject filtering
                    if subjects is not None and subject not in subjects:
                        continue

                    gst = int(group["gesture"][()])
                    rep_id = int(group["rep"][()])
                    rep_form = int(group["rep_form"][()])
                    emg = group["emg"][:] # (Samples, 8)

                    if emg.shape[0] == 0:
                        continue
                    
                    if not channel_last:
                        emg = emg.T

                    odh.data.append(emg)
                    odh.classes.append(np.ones((len(emg), 1), dtype=np.int64) * gst)
                    odh.subjects.append(np.ones((len(emg), 1), dtype=np.int64) * subject)
                    odh.reps.append(np.ones((len(emg), 1), dtype=np.int64) * rep_id)
                    odh.rep_forms.append(np.ones((len(emg), 1), dtype=np.int64) * rep_form)

        return odh

    def prepare_data(self, channel_last=True, subjects=None):
        processed_dir = self.dataset_folder + "_PROCESSED"

        if not os.path.exists(processed_dir):
            print(f"Processing dataset from {self.dataset_folder} to {processed_dir}...")
            process_dataset(self.dataset_folder, processed_dir)

        return self._get_odh(processed_dir, subjects, channel_last)