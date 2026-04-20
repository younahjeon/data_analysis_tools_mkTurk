import glob
import json
import os
from datetime import datetime

import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import stats


class BehaviorFileReader:
    """Lightweight reader for behavior json/txt files."""

    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, "r") as f:
            self.file = json.load(f)

    def read_all(self):
        return self.file

    def read_by_vars(self, variables):
        if isinstance(variables, list):
            result = {}
            for v in variables:
                result[v] = "not exist"
                for sub_dict in self.file:
                    if isinstance(sub_dict, dict):
                        if v in sub_dict:
                            result[v] = sub_dict[v]
                    else:
                        if v in self.file[sub_dict]:
                            result[v] = self.file[sub_dict][v]
            return result

        result = "not exist"
        for sub_dict in self.file:
            if isinstance(sub_dict, dict):
                if variables in sub_dict:
                    result = sub_dict[variables]
            else:
                if variables in self.file[sub_dict]:
                    result = self.file[sub_dict][variables]
        return result

    def get_nt(self):
        return len(self.read_by_vars(["Response"])["Response"])

    def get_perf(self):
        res = self.read_by_vars(["Response", "CorrectItem"])
        non_response = np.where(np.array(res["Response"]) == -1)
        valid_response = np.delete(np.array(res["Response"]), non_response)
        valid_correct = np.delete(np.array(res["CorrectItem"]), non_response)
        if len(valid_response) == 0:
            return np.nan
        return sum(valid_correct == valid_response) / len(valid_response)


class _PipelineHelpers:
    """Shared helper methods used by behavior processing pipelines."""

    @staticmethod
    def filter_trials(sequence, keep_inds):
        return [sequence[i] for i in keep_inds]

    @staticmethod
    def filter_trial_dict(data_dict, keep_inds):
        return {k: [v[i] for i in keep_inds] for k, v in data_dict.items()}

    @staticmethod
    def parse_file_date(file_path):
        try:
            if len(file_path.split("_")) > 2:
                suffix = file_path.split("_")[-1]
                return datetime.strptime(
                    file_path[0 : file_path.find(suffix) - 1], "%Y-%m-%dT%H_%M_%S"
                )
            f = file_path.split("/")[-1]
            return datetime.strptime(f[0 : f.find("_")], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None

    @staticmethod
    def build_bag_idx_from_scenes(scenes):
        bag_idx = [np.nan] * len(scenes)
        if len(scenes) == 0:
            return bag_idx
        bag_idx[0] = 0
        for i in range(1, len(scenes)):
            if scenes[i].split("/")[4] == scenes[i - 1].split("/")[4]:
                bag_idx[i] = bag_idx[i - 1]
            else:
                bag_idx[i] = bag_idx[i - 1] + 1
        return bag_idx

    @classmethod
    def resolve_scene_payload(cls, reader, scene_vars):
        if scene_vars["SampleScenes"] == "not exist":
            samplescenes = scene_vars["Ordered_Samplebag_Filenames"]
            testscenes = scene_vars["Ordered_Testbag_Filenames"]
            samplebagidx = cls.build_bag_idx_from_scenes(samplescenes)
            testbagidx = cls.build_bag_idx_from_scenes(testscenes)
        else:
            samplescenes = scene_vars["SampleScenes"]
            samplebagidx = scene_vars["SampleBagIdx"]
            testscenes = reader.read_by_vars(["TestScenes"])["TestScenes"]
            testbagidx = scene_vars["TestBagIdx"]
        return samplescenes, samplebagidx, testscenes, testbagidx

    @staticmethod
    def normalize_resdict(res_dict):
        if isinstance(res_dict["Sample"], dict):
            res_dict["Sample"] = res_dict["Sample"]["0"]
        if isinstance(res_dict["Test"], dict):
            test_values = list(res_dict["Test"].values())
            n_trials = len(test_values[0])
            n_choices = len(test_values)
            res_dict["Test"] = [
                [test_values[k][j] for k in range(n_choices)] for j in range(n_trials)
            ]
        return res_dict

    @staticmethod
    def infer_task_if_missing(meta):
        if meta["Task"] != "not exist":
            return
        if meta["ObjectGridIndex"] == []:
            meta["Task"] = "MTS"
        elif len(meta["ObjectGridIndex"]) == len(meta["ImageBagsSample"]):
            meta["Task"] = "SR"
        elif meta["RewardStage"] == 0:
            meta["Task"] = "Fixation"
        elif meta["SampleGridIndex"] == meta["TestGridIndex"][0] == meta["TestGridIndex"][1]:
            meta["Task"] = "SD"

    @staticmethod
    def extract_desired_tsequence(meta):
        if meta["TSequenceDesired"] != "not exist" and meta["TSequenceDesiredClip"] == "not exist":
            return meta["TSequenceDesired"][0]
        if meta["TSequenceDesiredClip"] != "not exist" and meta["TSequenceDesired"] == "not exist":
            return [value[0] for value in meta["TSequenceDesiredClip"].values()]
        return []

    @staticmethod
    def is_background_pairing_valid(webapp_url, imagebag, filedate):
        if filedate < datetime(2022, 7, 1, 18, 0, 0):
            return (
                (webapp_url == "https://mkturk.com/mkturk/index.html" and imagebag == "/mkturkfiles/imagebags/background_im/")
                or (webapp_url == "https://mkturk.com/mkturk/" and imagebag == "/mkturkfiles/imagebags/background_im/")
                or (webapp_url == "https://mkturk.com/mkturk-lab/index.html" and imagebag == "/mkturkfiles/scenebags/objectome3d/background_im_flipped/")
                or (webapp_url == "https://mkturk.com/mkturk-lab/" and imagebag == "/mkturkfiles/scenebags/objectome3d/background_im_flipped/")
            )
        return (
            (webapp_url == "https://mkturk.com/mkturk/" and imagebag == "/mkturkfiles/scenebags/objectome3d/background_im_flipped/")
            or (webapp_url == "https://mkturk.com/mkturk/index.html" and imagebag == "/mkturkfiles/scenebags/objectome3d/background_im_flipped/")
        )

    @classmethod
    def is_file_webapp_compatible(cls, meta, filedate):
        if meta["WebAppUrl"] == "not exist":
            return True
        for scene in meta["SampleScenes"]:
            imageidx = scene["IMAGES"]["imageidx"]
            imagebag = scene["IMAGES"]["imagebag"]
            if all(all(bb is None for bb in b) for b in imageidx):
                continue
            if imagebag not in (
                "/mkturkfiles/imagebags/background_im/",
                "/mkturkfiles/scenebags/objectome3d/background_im_flipped/",
            ):
                continue
            if not cls.is_background_pairing_valid(meta["WebAppUrl"], imagebag, filedate):
                return False
        return True

    @staticmethod
    def normalize_scenes_and_sample_on(meta):
        if meta["SampleON"] == "not exist":
            meta["SampleON"] = []
        if meta["SampleScenes"] == "not exist":
            meta["SampleScenes"] = meta["Ordered_Samplebag_Filenames"]
        elif meta["SampleON"] == [] and meta["KeepSampleON"] == 0:
            for scene in meta["SampleScenes"]:
                if "durationMS" not in scene:
                    continue
                duration = scene["durationMS"]
                meta["SampleON"].append(duration if isinstance(duration, int) else duration[0])
        if meta["TestScenes"] == "not exist":
            meta["TestScenes"] = meta["Ordered_Testbag_Filenames"]

    @staticmethod
    def find_matching_task(task, taskfiledate_dt, meta, filedate):
        for i, t in enumerate(task):
            dt = taskfiledate_dt[i] - filedate
            dt_days = dt.days + dt.seconds / (3600 * 24)
            if (
                t[0] == meta["ImageBagsSample"]
                and t[1] == meta["ImageBagsTest"]
                and t[4] == meta["Task"]
                and t[5] == meta["KeepSampleON"]
                and t[6] == meta["SampleON"]
                and dt_days > -14
                and t[7] == meta["Subject"]
            ):
                return i
        return -1

    @staticmethod
    def compute_choice_mappings(test_trials, responses, correct_items):
        newr = np.nan * np.ones(len(responses))
        newc = np.nan * np.ones(len(responses))
        newd = np.nan * np.ones(len(responses))
        for ind, (t, r) in enumerate(zip(test_trials, responses)):
            if isinstance(t, int):
                newr[ind] = r if r not in (-1, None) else -1
            else:
                if len(t) > 1 and r not in (-1, None):
                    newr[ind] = t[r]
                elif r not in (-1, None):
                    newr[ind] = r
                else:
                    newr[ind] = -1
        for ind, (t, c) in enumerate(zip(test_trials, correct_items)):
            if isinstance(t, int):
                newc[ind] = c
                newd[ind] = t
            else:
                if len(t) > 1:
                    newc[ind] = t[c]
                    t = np.array(t)
                    newd[ind] = t[np.arange(len(t)) != c]
                else:
                    newc[ind] = c
                    newd[ind] = t
        return newr, newc, newd

    @classmethod
    def compute_reaction_time(cls, res_dict, keep_inds, had_removed_trials):
        if isinstance(res_dict["FixationXYT"], dict):
            if had_removed_trials:
                res_dict["FixationXYT"] = cls.filter_trial_dict(res_dict["FixationXYT"], keep_inds)
            ft = res_dict["FixationXYT"]["2"]
        else:
            if had_removed_trials:
                res_dict["FixationXYT"] = cls.filter_trials(res_dict["FixationXYT"], keep_inds)
            ft = [t[2] for t in res_dict["FixationXYT"] if len(t) != 0]

        if len(ft) == len(res_dict["StartTime"]):
            ft_t = np.array(ft) - np.array(res_dict["StartTime"])
        else:
            ft_t = np.nan * np.ones(len(res_dict["StartTime"]))
        if res_dict["RewardStage"] == 0:
            return ft_t

        if isinstance(res_dict["ResponseXYT"], dict):
            if had_removed_trials:
                res_dict["ResponseXYT"] = cls.filter_trial_dict(res_dict["ResponseXYT"], keep_inds)
            respt = res_dict["ResponseXYT"]["2"]
        else:
            if had_removed_trials:
                res_dict["ResponseXYT"] = cls.filter_trials(res_dict["ResponseXYT"], keep_inds)
            respt = [t[2] for t in res_dict["ResponseXYT"] if len(t) > 0]

        if len(respt) == len(res_dict["StartTime"]) and not all(v is None for v in res_dict["StartTime"]) and not all(v is None for v in respt):
            respt_t = np.array(respt) - np.array(res_dict["StartTime"])
        else:
            respt_t = np.nan * np.ones(len(res_dict["StartTime"]))

        if isinstance(res_dict["TSequenceActualClip"], dict):
            if had_removed_trials:
                res_dict["TSequenceActualClip"] = cls.filter_trial_dict(res_dict["TSequenceActualClip"], keep_inds)
                res_dict["TSequenceDesiredClip"] = cls.filter_trial_dict(res_dict["TSequenceDesiredClip"], keep_inds)
            teston = res_dict["TSequenceActualClip"][str(len(res_dict["TSequenceActualClip"]) - 1)]
        else:
            if had_removed_trials:
                res_dict["TSequenceActual"] = cls.filter_trials(res_dict["TSequenceActual"], keep_inds)
                res_dict["TSequenceDesired"] = cls.filter_trials(res_dict["TSequenceDesired"], keep_inds)
            teston = [seq[len(seq) - 1] for seq in res_dict["TSequenceActual"]]

        return np.array(respt_t) - np.array(teston) - np.array(ft_t)

    @staticmethod
    def extract_tsequence_arrays(res_dict):
        if isinstance(res_dict["TSequenceActualClip"], dict):
            if len(res_dict["TSequenceActualClip"]) <= 1:
                return [], []
            num_trial = len(res_dict["TSequenceActualClip"]["0"])
            n_keys = len(res_dict["TSequenceActualClip"].keys())
            tseq = []
            tseq_d = []
            for n in range(num_trial):
                tseq_trial = np.nan * np.ones(n_keys)
                tseq_trial_d = np.nan * np.ones(n_keys)
                for ind, t in enumerate(res_dict["TSequenceActualClip"]):
                    tseq_trial[ind] = res_dict["TSequenceActualClip"][t][n]
                    tseq_trial_d[ind] = res_dict["TSequenceDesiredClip"][t][n]
                tseq.append(tseq_trial)
                tseq_d.append(tseq_trial_d)
            return tseq, tseq_d
        return res_dict["TSequenceActual"], res_dict["TSequenceDesired"]


class BehaviorProcessingPipeline:
    """Main behavior pipelines: file grouping + trial extraction."""

    @classmethod
    def sort_behavior_files(cls, folder_path):
        files_to_read = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".txt") or f.endswith(".json")]
        )

        task = []
        taskfileind = []
        taskfiledate = []
        taskfiledate_dt = []
        tasktsequence = []

        for ind, filename in enumerate(files_to_read):
            print(ind, filename)
            filedate = _PipelineHelpers.parse_file_date(filename)
            if filedate is None:
                continue

            file_path = os.path.join(folder_path, filename)
            meta = BehaviorFileReader(file_path).read_by_vars(
                [
                    "Agent",
                    "Subject",
                    "StressTest",
                    "Task",
                    "ObjectGridIndex",
                    "RewardStage",
                    "ImageBagsSample",
                    "ImageBagsTest",
                    "TSequenceDesired",
                    "TSequenceDesiredClip",
                    "KeepSampleON",
                    "SampleScenes",
                    "TestScenes",
                    "Ordered_Samplebag_Filenames",
                    "Ordered_Testbag_Filenames",
                    "SampleON",
                    "WebAppUrl",
                ]
            )

            if (meta["StressTest"] not in (0, "not exist")) or (meta["Agent"] == "SaveImages"):
                continue

            _PipelineHelpers.infer_task_if_missing(meta)
            tsequence = _PipelineHelpers.extract_desired_tsequence(meta)
            if not _PipelineHelpers.is_file_webapp_compatible(meta, filedate):
                continue

            _PipelineHelpers.normalize_scenes_and_sample_on(meta)
            file_task = (
                meta["ImageBagsSample"],
                meta["ImageBagsTest"],
                meta["SampleScenes"],
                meta["TestScenes"],
                meta["Task"],
                meta["KeepSampleON"],
                meta["SampleON"],
                meta["Subject"],
                meta["ObjectGridIndex"],
            )

            match_idx = _PipelineHelpers.find_matching_task(task, taskfiledate_dt, meta, filedate)
            if match_idx >= 0:
                taskfileind[match_idx].append(ind)
                taskfiledate[match_idx].append(filedate.isoformat())
                taskfiledate_dt[match_idx] = filedate
                tasktsequence[match_idx].append(tsequence)
            else:
                task.append(file_task)
                taskfileind.append([ind])
                taskfiledate.append([filedate.isoformat()])
                taskfiledate_dt.append(filedate)
                tasktsequence.append([tsequence])

        return task, tasktsequence, taskfileind, taskfiledate, files_to_read

    @classmethod
    def get_behavior_data(cls, files):
        sample = []
        response = []
        correctitem = []
        samplebagidx = []
        testbagidx = []
        objgrididx = []
        distractor = []
        samplescenes = []
        testscenes = []
        spatialbias = []
        perfbydate = []
        unique_date = []
        response_day = []
        correctitem_day = []
        num_trial_by_date = []
        reaction_time = []
        tseq_actual = []
        tseq_desired = []
        idx_files_survive = []

        for idx_f, file in enumerate(files):
            filedate = _PipelineHelpers.parse_file_date(file)
            if filedate is None:
                continue

            reader = BehaviorFileReader(file)
            scene_vars = reader.read_by_vars(
                [
                    "SampleScenes",
                    "Ordered_Samplebag_Filenames",
                    "Ordered_Testbag_Filenames",
                    "SampleBagIdx",
                    "TestBagIdx",
                    "ObjectGridIndex",
                    "TestGridIndex",
                ]
            )
            sample_sc, sample_idx, test_sc, test_idx = _PipelineHelpers.resolve_scene_payload(reader, scene_vars)
            obj_idx = scene_vars["ObjectGridIndex"]
            test_grid_idx = scene_vars["TestGridIndex"]
            if obj_idx == [] and test_grid_idx != []:
                obj_idx = test_grid_idx

            res = reader.read_by_vars(
                [
                    "Sample",
                    "Test",
                    "Response",
                    "CorrectItem",
                    "sequence",
                    "tsequence",
                    "SampleStartTime",
                    "StartTime",
                    "TSequenceActual",
                    "TSequenceActualClip",
                    "TSequenceDesired",
                    "TSequenceDesiredClip",
                    "ResponseXYT",
                    "FixationXYT",
                    "RewardStage",
                ]
            )
            res = _PipelineHelpers.normalize_resdict(res)

            valid_lengths = len(res["Response"]) == len(res["CorrectItem"]) == len(res["Sample"])
            if (not valid_lengths) or len(res["Sample"]) <= 10:
                continue

            keep_inds = [i for i, v in enumerate(res["Sample"]) if v is not None]
            had_removed = len(keep_inds) != len(res["Sample"])
            if had_removed:
                res["Sample"] = _PipelineHelpers.filter_trials(res["Sample"], keep_inds)
                res["Test"] = _PipelineHelpers.filter_trials(res["Test"], keep_inds)
                res["Response"] = _PipelineHelpers.filter_trials(res["Response"], keep_inds)
                res["CorrectItem"] = _PipelineHelpers.filter_trials(res["CorrectItem"], keep_inds)
                res["StartTime"] = _PipelineHelpers.filter_trials(res["StartTime"], keep_inds)

            newr, newc, newd = _PipelineHelpers.compute_choice_mappings(
                res["Test"], res["Response"], res["CorrectItem"]
            )
            rt = _PipelineHelpers.compute_reaction_time(res, keep_inds, had_removed)
            response_array = np.array(res["Response"])
            sb = [
                np.nansum(response_array == side) / len(res["Response"])
                for side in range(len(obj_idx))
            ]
            t_actual, t_desired = _PipelineHelpers.extract_tsequence_arrays(res)

            sample.append(res["Sample"])
            response.append(newr)
            distractor.append(newd)
            correctitem.append(newc)
            samplescenes.append(sample_sc)
            testscenes.append(test_sc)
            samplebagidx.append(sample_idx)
            testbagidx.append(test_idx)
            objgrididx.append(obj_idx)
            spatialbias.append(sb)
            reaction_time.append(rt)
            tseq_actual.append(t_actual)
            tseq_desired.append(t_desired)

            current_date = filedate.isoformat().split("T")[0]
            if idx_f == 0:
                response_day = newr
                correctitem_day = newc
            else:
                if current_date == f_date and idx_f != len(files) - 1:
                    response_day = np.concatenate((response_day, newr))
                    correctitem_day = np.concatenate((correctitem_day, newc))
                else:
                    perfbydate.append(
                        np.nansum(np.array(response_day) == np.array(correctitem_day))
                        / len(response_day)
                    )
                    num_trial_by_date.append(len(response_day))
                    unique_date.append(current_date if idx_f == len(files) - 1 else f_date)
                    if current_date != f_date:
                        response_day = newr
                        correctitem_day = newc

            f_date = current_date
            idx_files_survive.append(idx_f)
            if len(files) == 1:
                unique_date = [f_date]
                perfbydate.append(
                    np.nansum(np.array(response_day) == np.array(correctitem_day))
                    / len(response_day)
                )
                num_trial_by_date.append(len(response_day))

        if len(files) > 1 and len(idx_files_survive) == 1 and idx_files_survive[0] == 0:
            unique_date = [f_date]
            perfbydate.append(
                np.nansum(np.array(response_day) == np.array(correctitem_day)) / len(response_day)
            )
            num_trial_by_date.append(len(response_day))

        return (
            sample,
            response,
            correctitem,
            distractor,
            objgrididx,
            samplescenes,
            testscenes,
            samplebagidx,
            testbagidx,
            spatialbias,
            perfbydate,
            unique_date,
            num_trial_by_date,
            reaction_time,
            tseq_actual,
            tseq_desired,
        )


class BehaviorMetrics:
    """Metrics over processed behavior arrays."""

    @staticmethod
    def get_behavior_array(s, r, d, c, sidx, tidx, tasktype):
        s_uniqueobj, s_counts = np.unique(sidx, return_counts=True)
        t_uniqueobj, t_counts = np.unique(tidx, return_counts=True)
        n_dim1 = len(s_uniqueobj)
        n_dim2 = len(t_uniqueobj)
        n_dim3 = len(sidx)

        nc = np.nan * np.ones((n_dim1, n_dim2, n_dim3))
        nt = np.nan * np.ones((n_dim1, n_dim2, n_dim3))
        s_offsets = np.concatenate(([0], np.cumsum(s_counts)))
        t_offsets = np.concatenate(([0], np.cumsum(t_counts)))

        for obj_s in range(n_dim1):
            s_start, s_end = s_offsets[obj_s], s_offsets[obj_s + 1]
            sample_indices = np.arange(s_start, s_end, dtype=int)
            valid_target_objs = [obj_t for obj_t in range(n_dim2) if obj_t != obj_s]
            for obj_t in valid_target_objs:
                t_start, t_end = t_offsets[obj_t], t_offsets[obj_t + 1]
                img_obj_t = np.arange(t_start, t_end, dtype=int)
                for img_obj_s in sample_indices:
                    trial_ind = (s == img_obj_s) & (d == img_obj_t)
                    nc[obj_s, obj_t, img_obj_s] = np.nansum(r[trial_ind] == c[trial_ind])
                    nt[obj_s, obj_t, img_obj_s] = np.nansum(trial_ind)
        return nt, nc

    @staticmethod
    def get_i1(nt, nc, n_imgs):
        i2_hrs = nc / nt
        i2_hr_when_targ_was_distractor = np.nansum(nc, axis=-1) / np.nansum(nt, axis=-1)
        i2_fars = 1.0 - i2_hr_when_targ_was_distractor.T
        i2_z_hrs, i2_z_fars = stats.norm.ppf(i2_hrs), stats.norm.ppf(i2_fars)
        i2_dprimes = i2_z_hrs - i2_z_fars[:, :, None]
        i2_dprimes[np.isposinf(i2_z_hrs)], i2_dprimes[np.isneginf(i2_z_hrs)] = 4, -4

        i1_hrs = np.nansum(nc, axis=1) / np.nansum(nt, axis=1)
        i1_hrs_reshaped = np.nan * np.ones((i1_hrs.shape[0], n_imgs))
        for n in range(0, i1_hrs.shape[0]):
            if len(i1_hrs[n]) > n_imgs:
                i1_hrs_reshaped[n][0:n_imgs] = i1_hrs[n][n * n_imgs : (n + 1) * n_imgs]
            else:
                i1_hrs_reshaped[n][0:n_imgs] = i1_hrs[n][0:n_imgs]

        i1_hrs_when_targ_was_distractor = np.nansum(nc, axis=(0, 2)) / np.nansum(nt, axis=(0, 2))
        i1_fars = 1.0 - i1_hrs_when_targ_was_distractor
        i1_z_hrs, i1_z_fars = stats.norm.ppf(i1_hrs_reshaped), stats.norm.ppf(i1_fars)
        i1_z_fars[np.isneginf(i1_z_fars)] = 0
        i1_dprimes = i1_z_hrs - i1_z_fars[:, None]
        i1_dprimes[np.isposinf(i1_z_hrs)], i1_dprimes[np.isneginf(i1_z_hrs)] = 4, -4
        i1_dprimes[np.isneginf(i1_fars)] = i1_z_hrs[np.isneginf(i1_fars)]
        return i1_hrs, i1_fars, i1_dprimes, i2_hrs, i2_fars, i2_dprimes

    @staticmethod
    def i1_correlation(i1s, n_uniqueimg, remove_ceil):
        min_n_reps = 1000
        for i1 in i1s:
            min_n_reps = min(i1.shape[0], min_n_reps)

        i1_corrcoef = np.nan * np.ones((len(i1s), len(i1s)))
        i1_corrcoef_part = np.nan * np.ones((len(i1s), len(i1s), min_n_reps, 2))
        n_nans = np.nan * np.ones((len(i1s), len(i1s), min_n_reps, 2))
        for ind1, i1_1 in enumerate(i1s):
            for ind2, i1_2 in enumerate(i1s):
                corrcoef = []
                for w in range(min_n_reps):
                    x1, x2 = i1_1[w][0:n_uniqueimg], i1_1[w][n_uniqueimg : len(i1_1[w])]
                    y1, y2 = i1_2[w][0:n_uniqueimg], i1_2[w][n_uniqueimg : len(i1_2[w])]
                    if remove_ceil == 1:
                        for arr in (x1, x2, y1, y2):
                            arr[arr == 4] = np.nan
                            arr[arr == -4] = np.nan
                    num = (
                        ma.corrcoef(ma.masked_invalid(x1), ma.masked_invalid(y2))[0, 1]
                        + ma.corrcoef(ma.masked_invalid(x2), ma.masked_invalid(y1))[0, 1]
                    ) / 2
                    den = np.sqrt(
                        ma.corrcoef(ma.masked_invalid(x1), ma.masked_invalid(x2))[0, 1]
                        * ma.corrcoef(ma.masked_invalid(y1), ma.masked_invalid(y2))[0, 1]
                    )
                    n_nans[ind1, ind2, w] = [
                        sum(ma.masked_invalid(x1).mask) + sum(ma.masked_invalid(x2).mask),
                        sum(ma.masked_invalid(y1).mask) + sum(ma.masked_invalid(y2).mask),
                    ]
                    corrcoef.append(num / den)
                    i1_corrcoef_part[ind1, ind2, w] = [num, den]
                i1_corrcoef[ind1, ind2] = np.nanmean(corrcoef)
        return i1_corrcoef, i1_corrcoef_part, n_nans


class SceneAnalysis:
    """Scene/image utilities used by behavior analysis."""

    @staticmethod
    def get_longest_array(x):
        n = 0
        if type(x) != dict:
            if type(x) == list:
                return len(x)
            return 0
        for keys in x:
            if keys == "baseVertexInd":
                continue
            if type(x[keys]) == list:
                n_new = len(x[keys])
            elif type(x[keys]) == dict:
                n_new = SceneAnalysis.get_longest_array(x[keys])
            else:
                n_new = 0
            if n_new > n:
                n = n_new
        return n

    @staticmethod
    def get_latent_variables(scenefile):
        objname = list(scenefile["OBJECTS"].keys())[0]
        n_imgs = SceneAnalysis.get_longest_array(scenefile)

        rotX = np.nan * np.ones(n_imgs)
        rotY = np.nan * np.ones(n_imgs)
        rotZ = np.nan * np.ones(n_imgs)
        size = np.nan * np.ones(n_imgs)
        positionX = np.nan * np.ones(n_imgs)
        positionY = np.nan * np.ones(n_imgs)
        lightX = np.nan * np.ones(n_imgs)
        lightY = np.nan * np.ones(n_imgs)
        lightZ = np.nan * np.ones(n_imgs)
        cr = np.nan * np.ones(n_imgs)
        background = np.nan * np.ones(n_imgs)

        if len(scenefile["OBJECTS"][objname]["sizeInches"]) == 1:
            size = np.ones(n_imgs) * scenefile["OBJECTS"][objname]["sizeInches"][0]
        else:
            size = scenefile["OBJECTS"][objname]["sizeInches"]

        for axis_name, arr_name in [("x", "rotX"), ("y", "rotY"), ("z", "rotZ")]:
            vals = scenefile["OBJECTS"][objname]["rotationDegrees"][axis_name]
            data = np.ones(n_imgs) * vals[0] if len(vals) == 1 else vals
            if arr_name == "rotX":
                rotX = data
            elif arr_name == "rotY":
                rotY = data
            else:
                rotZ = data

        for axis_name, arr_name in [("x", "positionX"), ("y", "positionY")]:
            vals = scenefile["OBJECTS"][objname]["positionInches"][axis_name]
            data = np.ones(n_imgs) * vals[0] if len(vals) == 1 else vals
            if arr_name == "positionX":
                positionX = data
            else:
                positionY = data

        for axis_name, arr_name in [("x", "lightX"), ("y", "lightY"), ("z", "lightZ")]:
            vals = scenefile["LIGHTS"]["light00"]["position"][axis_name]
            data = np.ones(n_imgs) * vals[0] if len(vals) == 1 else vals
            if arr_name == "lightX":
                lightX = data
            elif arr_name == "lightY":
                lightY = data
            else:
                lightZ = data

        if len(scenefile["IMAGES"]["imageidx"]) == 1:
            background = np.ones(n_imgs) * scenefile["IMAGES"]["imageidx"][0]
        else:
            background = scenefile["IMAGES"]["imageidx"]

        if "OBJECTFILTERS" in scenefile:
            invert = scenefile["OBJECTFILTERS"]["invert"]
            if len(invert) == 1:
                cr = np.ones(n_imgs) * invert[0]
            elif len(invert) == 0:
                cr = np.ones(n_imgs) * 1
            else:
                cr = invert

        rotX = np.array(rotX)
        rotY = np.array(rotY)
        rotZ = np.array(rotZ)
        size = np.array(size)
        positionX = np.array(positionX)
        positionY = np.array(positionY)
        lightX = np.array(lightX)
        lightY = np.array(lightY)
        lightZ = np.array(lightZ)
        cr = [0 if n == "" else n for n in cr]
        background = [-1 if n == "" else n for n in background]
        if len(background) < n_imgs:
            background.extend([-1] * (n_imgs - len(background)))

        xnames = np.array(["p", "s", "vX", "vY", "vZ", "lX", "lY", "lZ", "cr", "background"])
        X = np.nan * np.ones((n_imgs, len(xnames)))
        pos = np.sqrt(np.square(positionX) + np.square(positionY))
        X[:, 0] = pos
        X[:, 1] = size
        X[:, 2] = rotX
        X[:, 3] = rotY
        X[:, 4] = rotZ
        X[:, 5] = lightX
        X[:, 6] = lightY
        X[:, 7] = lightZ
        X[:, 8] = cr
        X[:, 9] = background
        cols_to_keep = np.where(~np.all(np.isnan(X), axis=0))[0]
        return objname, X[:, cols_to_keep], xnames[cols_to_keep]


# Backward-compatible function aliases
def sortBehaviorfile(folder_path):
    return BehaviorProcessingPipeline.sort_behavior_files(folder_path)


def getBehaviorData(files):
    return BehaviorProcessingPipeline.get_behavior_data(files)


def getBehaviorArray(s, r, d, c, sidx, tidx, tasktype):
    return BehaviorMetrics.get_behavior_array(s, r, d, c, sidx, tidx, tasktype)


def getI1(nt, nc, n_imgs):
    return BehaviorMetrics.get_i1(nt, nc, n_imgs)


def I1correlation(I1s, n_uniqueimg, remove_ceil):
    return BehaviorMetrics.i1_correlation(I1s, n_uniqueimg, remove_ceil)

def getLongestArray(x):
    return SceneAnalysis.get_longest_array(x)


def get_lv(scenefile):
    return SceneAnalysis.get_latent_variables(scenefile)
