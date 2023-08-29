# Dataset Preparation

## Hidden Intention Dataset
1. Please download the dataset and annotations from [dataset provider](https://drive.google.com/drive/folders/1Yumk_BocNC7E64NekdPfT0nme5BX66g9?usp=drive_link).

2. Download the *frame list* from the following links: ([train](https://drive.google.com/drive/folders/10qFCFoYguZ1WWP93JzC4IJg8leEP3TQy?usp=drive_link), [val](https://drive.google.com/drive/folders/10qFCFoYguZ1WWP93JzC4IJg8leEP3TQy?usp=drive_link)).

3. Extract the frames at 30 FPS. (We used ffmpeg-4.1.3 with command
`ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"`
   in experiments.) Please put the frames in a structure consistent with the frame lists.


Please put all annotation json files and the frame lists in the same folder, and set `DATA.PATH_TO_DATA_DIR` to the path. Set `DATA.PATH_PREFIX` to be the path to the folder containing extracted frames.
