
Loading: /home/macs/Documents/rPPG-Controls/data/filterdesign_20260625_112442.csv

Loaded 1836 frames fs=30.00 Hz GT coverage=96% GT mean=103.1 BPM  
Cardiac band: [1.00, 3.23] Hz = [60, 194] BPM

  
── Section W: Noise Whiteness ──────────────────────────────────────  
Cardiac fundamental : 1.74 Hz (104 BPM)  
Ljung-Box (20 lags) : Q=6962.8 p=0.0000 → COLORED NOISE  
Eigenvalue CV : 2.909 (M=60) → non-uniform — colored noise confirmed  
────────────────────────────────────────────────────────────────────

  
==============================================================================================================  
Win(s) HamTight HamAdapt ElTight ElAdapt MUSIC ESPRIT  
(Welch/MAE) (Welch/MAE) (Welch/MAE) (Welch/MAE) (HamT/MAE) (HamT/MAE)  
==============================================================================================================

win=2s N=60 M=15 HQ=44/60 (73%) | LQ: SNR=14 Det=0 Skin=0 Lum=0  
2 16.98 16.18 20.48 17.66 17.86 25.00 [MAE BPM]  
46.7 50.0 46.7 51.7 36.7 26.7 [Acc %]  
HQ: HT=13.2 HA=13.6 ET=14.2 EA=12.0 MUS=16.1 ESP=19.8 [MAE BPM]

win=3s N=90 M=23 HQ=43/59 (73%) | LQ: SNR=12 Det=0 Skin=0 Lum=0  
3 13.94 13.12 18.53 15.71 9.99 36.90 [MAE BPM]  
52.5 50.8 47.5 54.2 66.1 37.3 [Acc %]  
HQ: HT=10.7 HA=11.5 ET=13.9 EA=13.9 MUS=8.1 ESP=29.8 [MAE BPM]

win=5s N=150 M=38 HQ=49/57 (86%) | LQ: SNR=7 Det=0 Skin=0 Lum=0  
5 8.80 9.58 13.96 12.38 8.80 45.73 [MAE BPM]  
71.9 68.4 63.2 68.4 75.4 3.5 [Acc %]  
HQ: HT=8.5 HA=8.8 ET=10.2 EA=8.8 MUS=7.9 ESP=45.3 [MAE BPM]

win=10s N=300 M=75 HQ=49/52 (94%) | LQ: SNR=2 Det=0 Skin=0 Lum=0  
10 3.74 3.77 6.85 9.31 3.59 13.37 [MAE BPM]  
92.3 88.5 82.7 73.1 98.1 57.7 [Acc %]  
HQ: HT=3.3 HA=3.5 ET=6.2 EA=8.6 MUS=3.7 ESP=12.0 [MAE BPM]

win=20s N=600 M=150 HQ=42/42 (100%) | LQ: SNR=0 Det=0 Skin=0 Lum=0  
20 2.11 2.11 2.10 4.27 2.02 4.62 [MAE BPM]  
100.0 100.0 100.0 85.7 100.0 85.7 [Acc %]  
HQ: HT=2.1 HA=2.1 ET=2.1 EA=4.3 MUS=2.0 ESP=4.6 [MAE BPM]

====================================================================================================  
MAE (BPM) and Acc (% within ±10 BPM) vs ground truth  
====================================================================================================