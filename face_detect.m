clear; close all; clc;

VID_PATH = '/home/macs/Documents/rPPG-Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/recording_2026-04-07T22-04-35Z.mov';
CSV_PATH = '/home/macs/Documents/rPPG-Controls/Measurement Data/DocBOT_2026-04-07_15-04-35/vitals.csv';

csv_data = readtable(CSV_PATH);
gt_time  = csv_data.offset_seconds;
gt_hr    = csv_data.heart_rate;
valid    = ~isnan(gt_hr);
gt_time  = gt_time(valid);  gt_hr = gt_hr(valid);
fprintf('GT: mean=%.1f  median=%.1f  range=[%d %d]  n=%d\n', ...
    mean(gt_hr), median(gt_hr), min(gt_hr), max(gt_hr), numel(gt_hr));

vid      = VideoReader(VID_PATH);
fs       = vid.FrameRate;
detector = vision.CascadeObjectDetector('MinSize', [80 80]);
tracker  = vision.PointTracker('MaxBidirectionalError', 2, 'NumPyramidLevels', 4);

% ── Initialise on first frame ─────────────────────────────────────────────
frame          = readFrame(vid);
bbox           = detectFace(detector, frame, vid.Width, vid.Height);
[pts, corners] = seedTracker(tracker, frame, bbox);

N          = round(vid.Duration * fs);
face_crops = cell(N, 1);
face_crops{1} = imcrop(frame, bbox);
k = 1;

% ── Track through all frames ──────────────────────────────────────────────
while hasFrame(vid)
    k     = k + 1;
    frame = readFrame(vid);
    [new_pts, ok] = step(tracker, frame);

    if sum(ok) >= 8 && mod(k, 30) ~= 0
        T       = estimateGeometricTransform2D(pts(ok,:), new_pts(ok,:), 'similarity');
        corners = transformPointsForward(T, corners);
        pts     = new_pts(ok, :);
        setPoints(tracker, pts);
        bbox    = corners2bbox(corners, vid.Width, vid.Height);
    else
        bbox           = detectFace(detector, frame, vid.Width, vid.Height);
        [pts, corners] = seedTracker(tracker, frame, bbox);
    end

    face_crops{k} = imcrop(frame, bbox);
end

face_crops = face_crops(1:k);
fprintf('Extracted %d face frames  fs=%.2f Hz\n', k, fs);

% ─────────────────────────────────────────────────────────────────────────
function bbox = detectFace(detector, frame, W, H)
    boxes = step(detector, frame);
    if isempty(boxes)
        bbox = [floor(W*0.25) floor(H*0.05) floor(W*0.50) floor(H*0.70)];
    else
        [~,i] = max(boxes(:,3) .* boxes(:,4));
        b     = boxes(i,:);
        bbox  = [max(b(1),1) max(b(2),1) min(b(3),W-b(1)) min(b(4),H-b(2))];
    end
end

function [pts, corners] = seedTracker(tracker, frame, bbox)
    feats = detectMinEigenFeatures(rgb2gray(frame), 'ROI', bbox, 'MinQuality', 0.05);
    pts   = feats.Location;
    if isLocked(tracker); setPoints(tracker, pts);
    else;                 initialize(tracker, pts, frame);
    end
    x = bbox(1); y = bbox(2); w = bbox(3); h = bbox(4);
    corners = [x,y; x+w,y; x+w,y+h; x,y+h];
end

function bbox = corners2bbox(corners, W, H)
    x1 = max(1, floor(min(corners(:,1))));
    y1 = max(1, floor(min(corners(:,2))));
    x2 = min(W,  ceil(max(corners(:,1))));
    y2 = min(H,  ceil(max(corners(:,2))));
    bbox = [x1 y1 x2-x1 y2-y1];
end
