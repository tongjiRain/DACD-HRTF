import os, glob
from torch.utils.data import Dataset
import sofa
from natsort import natsorted
import librosa

DATASET_PATH = "../data/"
TARGET_SAMPLING_RATE = 44100

class SOFADataset(Dataset):
    def __init__(self):
        super(SOFADataset, self).__init__()
        self.name = None
        self.sofa_dir = None  # a directory that include all the sofa files

    def _expand_basic_info(self):
        self.all_sofa_files = natsorted(self._get_all_sofa_files_from_dir()) # all the sofa files in the directory
        self.subject_IDs = [self._get_ID_from_sofa_path(path) for path in self.all_sofa_files]  # the id of the subjects
        self.num_of_subjects = len(self.subject_IDs)   # the length of the list of raw subject ids

    def _get_all_sofa_files_from_dir(self):
        raise NotImplementedError()

    def _get_ID_from_sofa_path(self, path):
        raise NotImplementedError()

    def _get_sofa_path_from_ID(self, subject_ID):
        raise NotImplementedError()

    def __len__(self):
        return self.num_of_subjects

    def __getitem__(self, idx):
        """
        :param idx: Here the idx is the idx of the ear, from it we can know the subject_idx and which_ear
        :return: locations and HRIRs
        """
        subject_idx = idx
        sofa_path = self.all_sofa_files[subject_idx]
        HRTF = sofa.Database.open(sofa_path)
        orig_sr = HRTF.Data.SamplingRate.get_values()[0]  # 初始采样率
        locations = HRTF.Source.Position.get_values(system="spherical") # (方位角,俯仰角, 距离半径）
        orig_hrirs = HRTF.Data.IR.get_values()  # 原始HRIR
        if orig_sr == TARGET_SAMPLING_RATE:
            return locations, orig_hrirs
        else:
            resample_hrirs = librosa.resample(orig_hrirs, orig_sr=orig_sr, target_sr=TARGET_SAMPLING_RATE)
            return locations, resample_hrirs


class ARI(SOFADataset):
    def __init__(self):
        super(ARI, self).__init__()
        self.name = "ari"
        self.sofa_dir = os.path.join(DATASET_PATH, "ARI/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "hrtf_nh*.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[1][2:].split(".")[0]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "hrtf_nh%s.sofa" % subject_ID)

class BiLi(SOFADataset):
    def __init__(self):
        super(BiLi, self).__init__()
        self.name = "bili"
        self.sofa_dir = os.path.join(DATASET_PATH, "BiLi/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "Test"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[1]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "IRC_%s_C_HRIR_96000.sofa" % subject_ID)

class CIPIC(SOFADataset):
    def __init__(self):
        super(CIPIC, self).__init__()
        self.name = "cipic"
        self.sofa_dir = os.path.join(DATASET_PATH, "CIPIC/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "subject_*.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[1][:3]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "subject_%03d.sofa" % int(subject_ID))


class Listen(SOFADataset):
    def __init__(self):
        super(Listen, self).__init__()
        self.name = "listen"
        self.sofa_dir = os.path.join(DATASET_PATH, "Listen/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "IRC_*_C_44100.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[1]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "IRC_%s_C_44100.sofa" % subject_ID)


class HUTUBS(SOFADataset):
    def __init__(self):
        super(HUTUBS, self).__init__()
        self.name = "hutubs"
        self.sofa_dir = os.path.join(DATASET_PATH, "HUTUBS/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "*HRIRs_measured.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[0][2:]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "pp%d_HRIRs_measured.sofa" % int(subject_ID))



class RIEC(SOFADataset):
    def __init__(self):
        super(RIEC, self).__init__()
        self.name = "riec"
        self.sofa_dir = os.path.join(DATASET_PATH, "RIEC/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "RIEC_hrir_subject_*.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[-1][:3]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "RIEC_hrir_subject_%s.sofa" % subject_ID)


class Prin3D3A(SOFADataset):
    def __init__(self):
        super(Prin3D3A, self).__init__()
        self.name = "3d3a"
        self.sofa_dir = os.path.join(DATASET_PATH, "Prin3D3A/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "Subject*_HRIRs.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[0][7:]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "Subject%s_HRIRs.sofa" % subject_ID)


class Crossmod(SOFADataset):
    def __init__(self):
        super(Crossmod, self).__init__()
        self.name = "crossmod"
        self.sofa_dir = os.path.join(DATASET_PATH, "Crossmod/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        return glob.glob(os.path.join(self.sofa_dir, "IRC_*_C_44100.sofa"))

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[1]

    def _get_sofa_path_from_ID(self, subject_ID):
        return os.path.join(self.sofa_dir, "IRC_%s_C_44100.sofa" % subject_ID)


class SONICOM(SOFADataset):
    def __init__(self):
        super(SONICOM, self).__init__()
        self.name = "sonicom"
        self.sofa_dir = os.path.join(DATASET_PATH, "SONICOM/sofa")
        self._expand_basic_info()

    def _get_all_sofa_files_from_dir(self):
        files_96k = glob.glob(os.path.join(self.sofa_dir, "P*_FreeFieldComp_96kHz.sofa"))
        files_48k = glob.glob(os.path.join(self.sofa_dir, "P*_FreeFieldCompMinPhase_48kHz.sofa"))
        return files_96k if files_96k else files_48k

    def _get_ID_from_sofa_path(self, path):
        return os.path.basename(path).split("_")[0][1:]

    def _get_sofa_path_from_ID(self, subject_ID):
        candidate_96k = os.path.join(self.sofa_dir, "P%s_FreeFieldComp_96kHz.sofa" % subject_ID)
        if os.path.exists(candidate_96k):
            return candidate_96k
        return os.path.join(self.sofa_dir, "P%s_FreeFieldCompMinPhase_48kHz.sofa" % subject_ID)





if __name__ == "__main__":
    master_dataset = CIPIC()
    location ,HRIR  = master_dataset.__getitem__(0)
    print(location.shape)
    print(HRIR.shape)






