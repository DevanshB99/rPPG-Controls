clear; close all; clc;

VID_PATH = '/home/macs/Documents/rPPG-Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/recording_2026-04-07T22-04-35Z.mov';

detector = vision.CascadeObjectDetector('MinSize', [80 80]);
player   = vision.VideoPlayer('Name', 'Face Detection Preview');

vid  = VideoReader(VID_PATH);
bbox = [];

while hasFrame(vid)
    frame = rot90(readFrame(vid), 3);   % 90° CW: VideoReader ignores .mov rotation metadata
    boxes = step(detector, frame);
    if ~isempty(boxes)
        [~,i] = max(boxes(:,3) .* boxes(:,4));
        bbox  = boxes(i,:);
    end
    if ~isempty(bbox)
        frame = insertShape(frame, 'Rectangle', bbox, 'Color', 'green', 'LineWidth', 4);
    end
    step(player, frame);
end
