#! /bin/bash
python -W ignore -O main.py --config configs/jspsr_r8_img.yml
python -W ignore -O main.py --config configs/jspsr_r8_img_msk.yml
python -W ignore -O main.py --config configs/jspsr_r3_img.yml
python -W ignore -O main.py --config configs/jspsr_r3_img_msk.yml
