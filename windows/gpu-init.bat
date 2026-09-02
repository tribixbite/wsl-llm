@echo off
REM ===================================================================
REM  GPU power caps for the 2x RTX 3090 box.  RUN ON WINDOWS, AS ADMIN.
REM
REM  This CANNOT be done from inside WSL2: the driver lives on the
REM  Windows side (GPU-PV), so `sudo nvidia-smi -pl` inside WSL fails
REM  with "Insufficient Permissions" even as root.  Verified 2026-09-02.
REM
REM  250 W is the measured sweet spot on a 3090 for Qwen3.8-27B decode:
REM    200 W ->  57.5 tok/s   (781 MHz)
REM    250 W ->  85.6 tok/s   (978 MHz)   <-- +49%
REM    280 W ->  no gain; hits 90 C in ~2 min and throttles back
REM  (14-minute sustained-load measurement, syv-ai/qwen38-27b-rtx3090#62)
REM
REM  Two cards at 250 W = 500 W of GPU.  The earlier hard resets on this
REM  box happened at 350 W x2 (~700 W).  500 W should be within budget,
REM  but watch for reset-under-load the first time both cards are busy.
REM ===================================================================

set PL=250

echo Setting persistence mode...
nvidia-smi -pm 1

echo Setting power limit to %PL% W on both GPUs...
nvidia-smi -i 0 -pl %PL%
nvidia-smi -i 1 -pl %PL%

echo.
echo Current state:
nvidia-smi --query-gpu=index,name,power.limit,temperature.gpu --format=csv

echo.
echo Done.  Verify from WSL with:
echo   nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
pause
