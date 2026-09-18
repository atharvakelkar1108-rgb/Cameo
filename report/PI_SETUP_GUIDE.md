# Raspberry Pi Zero 2 W — Setup & Hardware Test Guide

Follow this in order the moment the board arrives. Everything here is
already written and tested for syntax — you're just running it.

## 1. Flash the OS

Use the official **Raspberry Pi Imager** on your laptop:
- OS: **Raspberry Pi OS Lite (64-bit)** — no desktop, saves RAM
- Before writing: click the gear icon → enable SSH, set a username/password,
  and enter your WiFi details so it connects headlessly on first boot

## 2. First boot & connect

```bash
# find the Pi's IP address (check your router's admin page, or:)
ping raspberrypi.local

ssh <your-username>@<pi-ip-address>
```

## 3. Increase swap space (important — RAM is only 512MB)

```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# change: CONF_SWAPSIZE=100   to:   CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
free -h   # confirm swap now shows ~2GB
```
This won't make things fast, but it's the difference between a model
attempt failing outright (OOM-killed) vs. loading (slowly) so you get a
real measurement instead of a crash.

## 4. Install Python dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -r requirements-pi.txt   # NOT the main requirements.txt — this one is Pi-specific/minimal
```
This step is slow on a Pi Zero — expect 20-40 minutes for `torch` alone.
Let it run, don't interrupt it.

## 5. Transfer files from your PC

From your PC (PowerShell), copy the essentials — not the whole project,
just what's needed to run the test:

```powershell
scp scripts/09_pi_hardware_test.py <username>@<pi-ip>:~/
scp -r utils <username>@<pi-ip>:~/
scp requirements-pi.txt <username>@<pi-ip>:~/
```

If you want to test the quantized/pruned/optimized variants too (not just
baseline), also copy their `.pt` files — these are large (300MB-1GB each),
so start with just baseline first:
```powershell
scp results/quantized_model.pt <username>@<pi-ip>:~/
```

## 6. Run the hardware test

```bash
# on the Pi, with the venv activated
python3 09_pi_hardware_test.py --model baseline
```

**Based on your measured data, baseline is the recommended first attempt**
— it has the smallest RAM footprint (1277MB) of your four variants, even
though it's the largest file. Counterintuitive, but that's what your own
`--priority memory` selector run confirmed.

## 7. What to expect, and what to do with either outcome

- **If it succeeds:** record the load time, inference time, and RAM
  numbers it prints. This becomes your hardware validation section.
- **If it fails (OOM, killed process, or takes many minutes and you give
  up waiting):** that's still a complete, valid result. Record exactly
  what happened (the error message, or "still running after N minutes").
  Your report's conclusion becomes: *"Despite achieving up to 2.96x
  compression, the measured RAM footprint of all four model variants
  (1.3-2.8GB) exceeds the Pi Zero 2 W's 512MB RAM even with 2GB of swap
  space, demonstrating that file-size compression alone does not
  guarantee edge deployability — motivating the Resource-Aware Selector's
  RAM-footprint-first design."* That is a strong, honest, defensible
  conclusion — arguably more interesting than "it just worked."

## 8. If you want a live demo instead of just a logged result

Once `09_pi_hardware_test.py` runs successfully at least once, you can
also copy the full `webapp/` folder + `utils/` + one `results/*.json` set
to the Pi and run `python3 app.py` there — then browse to
`http://<pi-ip>:5000` from your phone or laptop for a live, on-device demo
during your viva. Only attempt this after confirming the basic hardware
test works — no point debugging the web app on top of an unresolved memory
issue.
