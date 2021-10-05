# RW_atmos
Solver of dispersion equations for RW-induced acoustic waves.

Python prerequisites:
- numpy,
- scipy,
- sympy,
- obspy,
- pyrocko,
- mpi4py.
```
pip3 install numpy scipy sympy obspy pyrocko mpi4py
```

Install `earthsr`:
```
make install
```

Make sure Python knows where earthsr is located.
In `utils.py`, set `earthsrExecutable` to point to the full path to the earthsr executable.