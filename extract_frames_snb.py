import os
import argparse
import cv2
from moviepy import *
from tqdm import tqdm
from multiprocessing import Pool
cv2.setNumThreads(0)

'''
This script extracts frames from SoccerNetv2 Ball Action Spotting dataset by introducing the path where the downloaded videos are (at 720p resolution), the path to
write the frames, the sample fps, and the number of workers to use. The script will create a folder for each video in the out_dir path and save the frames as .jpg files in
the desired resolution.

python extract_frames_snb.py --video_dir video_dir
        --out_dir out_dir
        --sample_fps 25 --num_workers 5
'''

FRAME_CORRECT_THRESHOLD = 1000

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Path to the downloaded videos')
    parser.add_argument('-o', '--out_dir',
                        help='Path to write frames. Dry run if None.')
    parser.add_argument('--sample_fps', type=int, default=25)
    parser.add_argument('-j', '--num_workers', type=int,
                        default=os.cpu_count() // 4)
    parser.add_argument('--target_height', type=int, default=448)
    parser.add_argument('--target_width', type=int, default=796)
    parser.add_argument('--original_resolution', type=str, default='720p')
    return parser.parse_args()


def get_duration(video_path):
    return moviepy.editor.VideoFileClip(video_path).duration


def worker(args):
    video_name, video_path, out_dir, sample_fps = args

    def get_stride(src_fps):
        if sample_fps <= 0:
            stride = 1
        else:
            stride = int(src_fps / sample_fps)
        return stride

    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)

    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    oh = TARGET_HEIGHT
    ow = TARGET_WIDTH

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    not_done = True
    while not_done:
        stride = get_stride(fps)
        est_out_fps = fps / stride
        print('{} -- effective fps: {} (stride: {})'.format(
            video_name, est_out_fps, stride))

        i = 0
        while True:
            ret, frame = vc.read()
            if not ret:
                # case when the video is corrupted
                if i != num_frames:
                    print('Failed to decode: {} -- {} / {}'.format(
                        video_path, i, num_frames))

                    if i + FRAME_CORRECT_THRESHOLD < num_frames:
                        print('Difference between expected and actual number of frames is too large')
                        
                not_done = False
                break

            if i % stride == 0:
                if frame.shape[0] != oh or frame.shape[1] != ow:
                    frame = cv2.resize(frame, (ow, oh))
                if out_dir is not None:
                    frame_path = os.path.join(
                        out_dir, 'frame{}.jpg'.format(i))
                    cv2.imwrite(frame_path, frame)
            i += 1
    vc.release()

    print('{} - done'.format(video_name))


def main(args):
    video_dir = args.video_dir
    out_dir = args.out_dir
    sample_fps = args.sample_fps
    num_workers = args.num_workers
    target_height = args.target_height
    target_width = args.target_width
    original_resolution = args.original_resolution

    global TARGET_HEIGHT
    TARGET_HEIGHT = target_height
    global TARGET_WIDTH
    TARGET_WIDTH = target_width

    out_dir = out_dir + str(TARGET_HEIGHT)

    worker_args = []
    for league in os.listdir(video_dir):
        print(league)
        
        if '.zip' in league:
            continue

        league_dir = os.path.join(video_dir, league)
        print(league_dir)
        for season in os.listdir(league_dir):
            season_dir = os.path.join(league_dir, season)
            print(season_dir)
            for game in os.listdir(season_dir):
                game_dir = os.path.join(season_dir, game)
                print(game_dir)
                for video_file in os.listdir(game_dir):
                    
                    print(video_file)
                    print(original_resolution)
                    print("is video?: " + str(video_file.endswith(original_resolution + '.mp4') | video_file.endswith(original_resolution + '.mkv')))
                    if (video_file.endswith(original_resolution + '.mp4') | video_file.endswith(original_resolution + '.mkv')):
                        print(video_file)
                        worker_args.append((
                            os.path.join(league, season, game, video_file),
                            os.path.join(game_dir, video_file),
                            os.path.join(
                                out_dir, league, season, game
                            ) if out_dir else None,
                            sample_fps
                        ))

    print(worker_args)
    with Pool(num_workers) as p:
        for _ in tqdm(p.imap_unordered(worker, worker_args),
                    total=len(worker_args)):
            pass
    print('Done!')


if __name__ == '__main__':
    args = get_args()
    main(args)