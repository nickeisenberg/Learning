'''
--------------------some notes--------------------

- rfft, fft and cross_correlation all seem to work the same when len(Ref1), which is the
same as len(Pat1), is an even number.

- fft and cross correlation always coincide.

- It seems that regardless of whether len(Ref1) is even or odd, the argmax of rfft, fft and
cross_correlation are all the same.

- Everything should be thought of in terms of the cross correlation. The scores of the cross
correlation are essentially the shifted inner products. The biggest score value will then
correspond to the highest shift one needs to make so the two signals are the most
correlated.

- There is a pattern and then there is a reference. The score indices will then tell us
how much we need to move the pattern and in what direction so that the pattern and the
reference will be the most correlated. The fft and rfft trick does the same thing which follows
from standard properties of the cross correlation which can be found on wikipedia.

- If the pattern signal and reference signal are the same signal but just delayed, then
the score indicie will tell you the exact delay. The follows immediately because the score
indicie is the shift one needs to make to the pattern to give the highest correlation. If the
signals are the same but delayed then the shift is the amount of time the signals are delayed.

- I am unsure of the meaning for the scores_compares...
'''

import numpy as np
from correlation import circ_cross_corr
from copy import copy

pat = np.random.randn(5)
ref1 = copy(pat[1:4])
ref1 = np.concatenate([ref1, [0, 0]])
pat[0] = 0
pat[-1] = 0
Ref1 = np.hstack([np.zeros(len(pat)), ref1])
Pat1 = np.hstack([pat, np.zeros(len(ref1))])

# ref2 = [3, 1, 6, 9, 88, 11]
# Ref2 = np.hstack([np.zeros(len(pat)), ref2])
# Pat2 = np.hstack([pat, np.zeros(len(ref2))])
# scores2 = np.fft.irfft(np.conj(np.fft.rfft(Pat2)) * np.fft.rfft(Ref2))
# inside2 = np.conj(np.fft.rfft(Pat2)) * np.fft.rfft(Ref2)

scores1_rfft = np.round(np.fft.irfft(np.conj(np.fft.rfft(Pat1)) * np.fft.rfft(Ref1)).real, 3)

scores_compares = np.concatenate([np.linspace(0.0, len(Pat1) - 1, num=len(Pat1)),
                                  len(Pat1) * np.ones(len(Ref1) - len(Pat1) + 1),
                                  np.linspace(len(Pat1) - 1, 1.0, num=len(Ref1)-1)])

print('--------------------------------------------------')
print(f'Correlation for:\nPattern to look for : {pat}\nReference to look in : {ref1}\n')
print(f'Pat1 : {Pat1}\nRef1 : {Ref1}\n')
scores_indicies = np.arange(len(scores1_rfft), dtype=int) - len(pat)
print(f'Scores_indicies : {scores_indicies}\n')
print(f'Scores_compares : {scores_compares}')

print('-------------------------')
print(f'Using irfft(conj(rfft(Pat1)) * rfft(Ref1))...')
print(f'Scores:\n{scores1_rfft}')
arg_rfft = np.argmax(scores1_rfft)
print(f'Arg max : {arg_rfft}')
print(f'Scores_indicies[arg_rfft] : {scores_indicies[arg_rfft]}')
print('-------------------------')

print(f'Using ifft(conj(fft(Pat1)) * fft(Ref1))...')
scores1_rfft_fft = np.round(np.fft.ifft(np.conj(np.fft.fft(Pat1)) * np.fft.fft(Ref1)).real, 3)
print(f'Scores\n{scores1_rfft_fft}')
arg_fft = np.argmax(scores1_rfft_fft)
print(f'Arg max : {arg_fft}')
print(f'Scores_indicies[arg_fft] : {scores_indicies[arg_fft]}')
print('-------------------------')

print('Using the circ_cross_corr function...')
scores1_cor = np.round(np.array([circ_cross_corr(Pat1, Ref1, i) for i in range(len(Pat1))]), 3)
# scores1_cor = np.round(np.array([cross_correlation(Pat1, Ref1, i) for i in np.array([-3, -2, -1, 0, 1])]), 3)
print(f'Scores :\n{scores1_cor}')
arg_cor = np.argmax(scores1_cor)
print(f'Arg max : {arg_cor}')
print(f'Scores_indicies[arg_cor] : {scores_indicies[arg_cor]}')
print('-------------------------')


