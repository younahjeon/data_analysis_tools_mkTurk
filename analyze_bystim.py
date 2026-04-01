from natsort import os_sorted
import pickle
import pathlib as Path
import os
import sys
import numpy as np
from utils_ephys import * 
from utils_meta import * 
from joblib import Parallel, delayed


def process_channel_data(n_chan, data_path,MUA_dir, data_dict_path, stim_info_path, save_out_path, plot_save_out_path):
    config = Config()
    try:
        # get baseline
        ch_psth_bl, ch_psth_bl_meta,ch_psth_bl_stim = get_data_bl(n_chan, MUA_dir, data_dict_path,stim_info_path)

        pickle.dump(ch_psth_bl, open(save_out_path /'ch{:0>3d}_psth_bl'.format(n_chan) ,'wb'), protocol = 2)
        pickle.dump(ch_psth_bl_stim, open(save_out_path /'ch{:0>3d}_psth_bl_stim'.format(n_chan) ,'wb'), protocol = 2)
        # get spike times, peak amplitude, waveform, and psth per stimulus 

        ch_pk_stim, ch_wf_stim, ch_st_stim, ch_sl_stim, ch_ind_stim,ch_psth_stim,ch_psth_stim_meta = \
        get_data_bystim(n_chan, MUA_dir, stim_info_path, binwidth_psth= 0.01)
        
        pickle.dump(ch_pk_stim, open(save_out_path / 'ch{:0>3d}_pk_stim'.format(n_chan),'wb'), protocol = 2)
        pickle.dump(ch_wf_stim, open(save_out_path / 'ch{:0>3d}_wf_stim'.format(n_chan),'wb'), protocol = 2)
        pickle.dump(ch_st_stim, open(save_out_path / 'ch{:0>3d}_st_stim'.format(n_chan),'wb'), protocol = 2)
        pickle.dump(ch_sl_stim, open(save_out_path / 'ch{:0>3d}_sl_stim'.format(n_chan),'wb'), protocol = 2)
        pickle.dump(ch_ind_stim, open(save_out_path / 'ch{:0>3d}_ind_stim'.format(n_chan),'wb'), protocol = 2)
        pickle.dump(ch_psth_stim, open(save_out_path /'ch{:0>3d}_psth_stim'.format(n_chan),'wb'), protocol = 2)
        pickle.dump(ch_psth_stim_meta, open(save_out_path /'psth_stim_meta'.format(n_chan),'wb'), protocol = 2)

        # get mean psth per stimulus
        # get the longest length 
        psth_len= 0
        for stim in ch_psth_stim:
            psth_len = np.nanmax((psth_len,ch_psth_stim[stim].shape[1]))

        mean_psth = np.nan * np.ones((len(ch_psth_stim), psth_len))
        mean_fr = np.nan * np.ones((len(ch_psth_stim)))
        sd_fr = np.nan * np.ones((len(ch_psth_stim)))
        for n,stim in enumerate(ch_psth_stim):
            mean_psth[n,0:ch_psth_stim[stim].shape[1]] = np.nanmean(ch_psth_stim[stim],axis = 0)
            mean_fr[n] = np.nanmean(ch_psth_stim[stim][:,config.t_early_bin[0]:config.t_late_bin[1]])
            sd_fr[n] = np.nanstd(ch_psth_stim[stim][:,config.t_early_bin[0]:config.t_late_bin[1]])

        np.save(save_out_path / 'ch{:0>3d}_mean_psth.npy'.format(n_chan),mean_psth)
        np.savez(save_out_path / 'ch{:0>3d}_mean_sd_fr.npz'.format(n_chan),mean_fr = mean_fr, sd_fr = sd_fr)

        # trial sd per stimulus

        # get psth per scenefile

        stim_list_sess = list(ch_psth_stim_meta.keys())

        all_scenefile = []
        for stim in stim_list_sess:
            all_scenefile.extend(ch_psth_stim_meta[stim]['scenefile'])

        unique_scenefile = np.unique(all_scenefile)

        psth_byscenefile = dict.fromkeys(unique_scenefile) 
        psth_byscenefile_meta = dict.fromkeys(unique_scenefile)

        for s in unique_scenefile:
            psth_new = []
            psth_byscenefile_meta[s] = dict()
            psth_byscenefile_meta[s]['stim_ids'] = []
            psth_byscenefile_meta[s]['ind'] = []
            psth_byscenefile_meta[s]['n_trials_stim'] = []
            n_trials = 0
            for stim_ind,stim in enumerate(stim_list_sess):

                if s in ch_psth_stim_meta[stim]['scenefile']:
                    # find trials where the stimulus appeared within that scenefile 
                    tr_idx = np.where(np.array(ch_psth_stim_meta[stim]['scenefile']) == s)[0]
                    # baseline correct it! 20240319
                    bl = np.nanmean(ch_psth_bl_stim[stim][tr_idx,:])
                    #psth_new.append(ch_psth_stim[stim] - bl)
                    stim_dur = ch_psth_stim_meta[stim]['stim_dur']
                    psth_new.append(ch_psth_stim[stim][tr_idx,:])
                    psth_byscenefile_meta[s]['ind'].append(stim_ind)
                    psth_byscenefile_meta[s]['stim_ids'].append(stim)
                    t_before = ch_psth_stim_meta[stim]['t_before']
                    t_after = ch_psth_stim_meta[stim]['t_after']
                    psth_bins = ch_psth_stim_meta[stim]['psth_bins']
                    
                    binwidth = ch_psth_stim_meta[stim]['binwidth']
                    n_trials += ch_psth_stim_meta[stim]['n_trials']
                    psth_byscenefile_meta[s]['n_trials_stim'].append(len(tr_idx))


            psth_new = np.vstack(psth_new)
            psth_byscenefile[s]= psth_new
            psth_byscenefile_meta[s]['stim_dur'] = stim_dur
            psth_byscenefile_meta[s]['t_before'] = t_before
            psth_byscenefile_meta[s]['t_after'] = t_after
            psth_byscenefile_meta[s]['psth_bins'] = psth_bins
            psth_byscenefile_meta[s]['binwidth'] = binwidth
            psth_byscenefile_meta[s]['n_trials'] = n_trials

        pickle.dump(psth_byscenefile, open(save_out_path /'ch{:0>3d}_psth_scenefile'.format(n_chan),'wb'), protocol = 2)
        pickle.dump(psth_byscenefile_meta, open(save_out_path /'psth_scenefile_meta','wb'), protocol = 2)

        # scenefile psth 
        try: 
            chanmap, _ = get_chanmap(data_path)
        except:
            chanmap = np.load(save_out_path / 'chanmap.npy')

        gen_psth_byscenefile(n_chan,plot_save_out_path, psth_byscenefile, psth_byscenefile_meta,chanmap)
        print(f"Successfully processed channel {n_chan:03d}")
        return n_chan, True, None
    except Exception as e:
        print(f"Error processing channel {n_chan:03d}: {str(e)}")
        return n_chan, False, str(e)
    
def process_single_directory(data_path, MUA_dir, data_dict_path, stim_info_path, save_out_path, plot_save_out_path, 
                             channel_list,n_jobs):
    
    # Process channels in parallel
    print(f"\nProcessing {len(channel_list)} channels in parallel with {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_channel_data)(
            n_chan, data_path, MUA_dir, data_dict_path, stim_info_path, save_out_path, plot_save_out_path
        ) for n_chan in channel_list
    )
    
    return results

def main():
    n_chan = int(sys.argv[1])
    monkey = sys.argv[2]
    date = sys.argv[3]
    engram_path, base_data_path, base_save_out_path = setup_paths()

    data_path_list, save_out_path_list, plot_save_out_path_list = init_dirs(base_data_path, monkey, date, base_save_out_path)

    for n, (data_path, save_out_path, plot_save_out_path) in enumerate(zip(data_path_list, save_out_path_list, plot_save_out_path_list)):
        MUA_dir = data_path / Path.Path('MUA_4SD')
        print(data_path)
        if not MUA_dir.exists():
            print('MUA path doesn''t exist')
            if n == len(data_path_list)-1:
                sys.exit()
            else:
                continue

        if not save_out_path.exists():
            print('save out path doesn''t exist')
            if n == len(data_path_list) - 1:
                sys.exit()
            else:
                continue

        try: 
            stim_info_path= os_sorted(save_out_path.glob('stim_info_sess'))[0]
            print(stim_info_path)
            print('\nstim info sess found: ' + str(stim_info_path.exists()))
        except:
            print('no stim_info_sess found')
            if n == len(data_path_list) -1 :
                sys.exit()
            else:
                continue

        try:
            data_dict_path = os_sorted(save_out_path.glob('data_dict_' + monkey + '*'))[0]
        except:
            print('no data dict found')
            if n == len(data_path_list) -1 :
                sys.exit()
            else:
                continue

        if not plot_save_out_path.exists():
            os.makedirs(plot_save_out_path,exist_ok= True)
        ##############################################################################################################################################
        process_channel_data(n_chan, data_path, MUA_dir, data_dict_path, stim_info_path, save_out_path, plot_save_out_path)
    

if __name__ == "__main__":
    main()