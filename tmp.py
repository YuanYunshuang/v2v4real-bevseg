import glob, shutil, os, tqdm

in_dir = "/koko/OPV2V/additional"
out_dir = "/koko/OPV2V/augmented"

# img_files = glob.glob(os.path.join(in_dir, '*/*/*/*.png'))
#
# for f in tqdm.tqdm(img_files):
#     try:
#         dst = f.replace('additional', 'augmented').replace('_road', '')
#         shutil.copy2(f, dst)
#     except:
#         os.makedirs(os.path.dirname(dst))
#         shutil.copy(f, dst)

# yaml_files = glob.glob(os.path.join(in_dir, '*/*/*/*.yaml')) + glob.glob(os.path.join(in_dir, '*/*/*.yaml'))
#
# for f in tqdm.tqdm(yaml_files):
#     try:
#         dst = f.replace('additional', 'augmented')
#         shutil.copy2(f, dst)
#     except:
#         os.makedirs(os.path.dirname(dst))
#         shutil.copy(f, dst)


bin_files = glob.glob(os.path.join(in_dir, '*/*/*/*_semantic_lidarcenter.bin'))

for f in tqdm.tqdm(bin_files):
    try:
        dst = f.replace('additional', 'augmented').replace('_semantic_lidarcenter', '')
        shutil.copy2(f, dst)
    except:
        os.makedirs(os.path.dirname(dst))
        shutil.copy(f, dst)