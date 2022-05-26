# Data Preparation


- The pre-processing of **Something-Something-V2** follows [VideoMAE](https://github.com/MCG-NJU/VideoMAE), 
which can be summarized into 3 steps:

  1. Download the dataset from [official website](https://developer.qualcomm.com/software/ai-datasets/something-something).

  2. Preprocess the dataset by changing the video extension from `webm` to `.mp4` with the **original** height of **240px**.. You can simply run `ffmpeg -i [input.webm] -c:v libx264 [output.mp4]`.

  3. Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations). The annotation usually includes `train.csv`, `val.csv` and `test.csv` ( here `test.csv` is the same as `val.csv`). The format of `*.csv` file is like:

     ```
     dataset_root/video_1.mp4  label_1
     dataset_root/video_2.mp4  label_2
     dataset_root/video_3.mp4  label_3
     ...
     dataset_root/video_N.mp4  label_N
     ```